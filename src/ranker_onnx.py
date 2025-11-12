from typing import List, Tuple, Dict
import os
import numpy as np

# Optional imports
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None


class PseudoLikelihoodRanker:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        onnx_path: str = None,
        device: str = "cpu",
        max_length: int = None,
    ):
        self.model_name = model_name
        self.device = device
        # configurable latency guards
        self.max_length = max_length or int(os.environ.get("MAX_SEQ_LEN", 64))
        self.max_candidates = int(os.environ.get("MAX_CANDS", 6))

        self.tokenizer = None
        self.onnx = None
        self.torch_model = None

        # Prefer ONNX if provided and runtime is available
        if onnx_path and ort is not None:
            self._init_onnx(onnx_path)
        elif AutoTokenizer is not None and AutoModelForMaskedLM is not None:
            self._init_torch()
        else:
            raise RuntimeError(
                "Neither onnxruntime nor transformers/torch are available. Please install requirements."
            )

    # ---------------------- initialization ----------------------
    def _init_onnx(self, onnx_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        providers = ["CPUExecutionProvider"]
        self.onnx = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=so
        )

    def _init_torch(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.torch_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.torch_model.eval()
        self.torch_model.to(self.device)

    # ---------------------- helpers ----------------------
    def _masked_positions(self, input_ids: np.ndarray, attn: np.ndarray):
        mask_id = self.tokenizer.mask_token_id
        seq = input_ids[0]
        L = int(attn[0].sum())
        positions = list(range(1, L - 1))
        batch = np.repeat(seq[None, :], len(positions), axis=0)
        for i, pos in enumerate(positions):
            batch[i, pos] = mask_id
        batch_attn = np.repeat(attn, len(positions), axis=0)
        return batch, batch_attn, np.array(positions, dtype=np.int64)

    # ---------------------- scoring ----------------------
    def _score_with_onnx(self, text: str) -> float:
        """Moderate-cost pseudo-likelihood scoring (~20ms typical)."""
        toks = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]
        seq = input_ids[0]
        L = int(attn[0].sum())
        mask_id = self.tokenizer.mask_token_id

        # score roughly every 2nd token to trade accuracy vs speed
        positions = list(range(1, L - 1, 2))
        total = 0.0
        for pos in positions:
            masked = seq.copy()
            token_id = int(masked[pos])
            masked[pos] = mask_id
            ort_inputs = {
                "input_ids": masked[None, :].astype(np.int64),
                "attention_mask": attn.astype(np.int64),
            }
            logits = self.onnx.run(None, ort_inputs)[0]
            logits_pos = logits[0, pos, :]
            # log-softmax manually for stability
            m = logits_pos.max()
            log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum())
            total += float(log_probs[token_id])
        # normalize by number of scored positions
        return total / max(1, len(positions))

    def _score_with_torch(self, text: str) -> float:
        toks = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]
        seq = input_ids[0]
        L = int(attn.sum())
        positions = list(range(1, L - 1))
        batch = seq.unsqueeze(0).repeat(len(positions), 1)
        for i, pos in enumerate(positions):
            batch[i, pos] = self.tokenizer.mask_token_id
        batch_attn = attn.repeat(len(positions), 1)
        with torch.no_grad():
            out = self.torch_model(
                input_ids=batch, attention_mask=batch_attn
            ).logits
            orig = seq.unsqueeze(0).repeat(len(positions), 1)
            rows = torch.arange(len(positions))
            cols = torch.tensor(positions)
            token_ids = orig[rows, cols]
            logits_pos = out[rows, cols, :]
            log_probs = logits_pos.log_softmax(dim=-1)
            picked = log_probs[torch.arange(len(rows)), token_ids]
        return float(picked.sum().item())

    def score(self, sentences: List[str]) -> List[float]:
        if self.onnx is not None:
            return [self._score_with_onnx(s) for s in sentences]
        return [self._score_with_torch(s) for s in sentences]

    # ---------------------- ranking ----------------------
    def choose_best(self, candidates: List[Tuple[str, Dict]]) -> Tuple[str, Dict]:
        if not candidates:
            return "", {}

        # prefer validated (emails/numbers) but rescore them once for quality
        valids = [(t, m) for t, m in candidates if m.get("validated")]
        if valids:
            texts = [t for t, _ in valids]
            scores = self.score(texts)
            best_idx = int(np.argmax(scores))
            s, m = valids[best_idx]
            m = {**m, "ranker": "validated_rescore", "score": scores[best_idx]}
            return s, m

        # otherwise same fallback logic
        uniq, seen = [], set()
        for t, m in sorted(candidates, key=lambda x: len(x[0])):
            if t not in seen:
                uniq.append((t, m))
                seen.add(t)
            if len(uniq) >= self.max_candidates:
                break

        texts = [t for t, _ in uniq]
        scores = self.score(texts)
        best_idx = int(np.argmax(scores))
        best_text, best_meta = uniq[best_idx]
        best_meta = {**best_meta, "ranker": "pl_score", "score": scores[best_idx]}
        return best_text, best_meta

