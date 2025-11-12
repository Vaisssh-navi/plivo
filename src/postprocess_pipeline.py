import json
from typing import Dict, List
from .rules import generate_candidates
from .ranker_onnx import PseudoLikelihoodRanker


class PostProcessor:
    def __init__(self, names_lex_path: str, onnx_model_path: str = None, device: str = "cpu", max_length: int = 64):
        # Load lexicon of known Indian names
        with open(names_lex_path, 'r', encoding='utf-8') as f:
            self.names_lex = [x.strip() for x in f if x.strip()]
        # Initialize fast pseudo-likelihood ranker
        self.ranker = PseudoLikelihoodRanker(
            onnx_path=onnx_model_path, device=device, max_length=max_length
        )

    def process_one(self, text: str) -> str:
        """Generate corrected text for a single utterance."""
        if not text or not text.strip():
            return text

        cands = generate_candidates(text, self.names_lex)
        best_text, meta = self.ranker.choose_best(cands)

        # --- Minimal punctuation rule ---
        lower = best_text.strip().lower()
        if not lower:
            return best_text
        if not lower.endswith(('.', '?', '!', ',')):
            first = lower.split()[0] if lower.split() else ''
            if first in (
                'can', 'shall', 'will', 'could', 'would',
                'is', 'are', 'do', 'does', 'did', 'should',
                'hey', 'hello'
            ):
                best_text = best_text.rstrip() + '?'
            else:
                best_text = best_text.rstrip() + '.'
        return best_text

    def process_file(self, input_path: str, output_path: str):
        """Run the post-processor on a JSONL file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]

        out = []
        for r in rows:
            pred = self.process_one(r["text"])
            out.append({"id": r["id"], "text": pred})

        with open(output_path, 'w', encoding='utf-8') as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")


def run_file(
    input_path: str,
    output_path: str,
    names_lex_path: str,
    onnx_model_path: str = None,
    device: str = "cpu",
    max_length: int = 64,
):
    """Convenience wrapper for CLI / score.sh usage."""
    pp = PostProcessor(
        names_lex_path,
        onnx_model_path=onnx_model_path,
        device=device,
        max_length=max_length,
    )
    pp.process_file(input_path, output_path)
