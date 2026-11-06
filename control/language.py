"""Resident Hermes intent parser; no motor, camera or subprocess access."""
from __future__ import annotations

import json
from pathlib import Path
import re
import runpy
import time
from .targets import normalize_task


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate model JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite model JSON value: " + value)


def validate_model_task(raw):
    """Require the complete model contract and return executable parameters."""
    task = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_nonfinite)
    keys = {
        "push_distance": {"intent", "color", "direction", "distance_mm"},
        "push_to_end": {"intent", "color", "direction"},
        "push_round_trip": {"intent", "color", "round_trips"},
    }
    if (not isinstance(task, dict) or not isinstance(task.get("intent"), str) or
            task["intent"] not in keys):
        raise ValueError("unknown model intent")
    if set(task) != keys[task["intent"]]:
        raise ValueError("model task has missing or extra keys")
    if (task["intent"] == "push_to_end" and
            task["direction"] not in ("+x", "-x", "other", None)):
        raise ValueError("invalid model push_to_end direction")
    return normalize_task(task)


class ResidentModel:
    """Load tokenizer, weights and their training prompt once at startup."""

    def __init__(self, package_path):
        package = Path(package_path).resolve()
        entry = package / "mini_llama.py"
        if not entry.is_file():
            raise ValueError("model package missing mini_llama.py: " + str(package))
        model_dir = package / "model"
        metadata = json.loads((model_dir / "train_meta.json").read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_keys)
        self.system = metadata["system"]
        if not isinstance(self.system, str) or not self.system.strip():
            raise ValueError("model training system prompt is empty")
        hermes = runpy.run_path(str(entry))
        # Always use the bundled PureBPE, including on development computers
        # with tokenizers installed, so the board follows the same path.
        self.tokenizer = hermes["Tok"](model_dir, backend="pure")
        self.model, _, info = hermes["load_model"](model_dir)
        if info["missing"] or info["unexpected"]:
            raise ValueError("model weights do not match configuration")
        self._tensor = hermes["torch"].tensor

    def infer(self, text):
        if not isinstance(text, str):
            raise ValueError("natural-language instruction must be text")
        text = text.lstrip("\ufeff").strip()
        if not text:
            raise ValueError("empty natural-language instruction")
        # This model misreads negative commands; explicit refusals never reach it.
        if (any(word in text for word in ("不要", "不许", "禁止", "停止", "取消", "不需要")) or
                re.search(r"(?:^|[，。！？\s])(?:请|麻烦|劳驾)?别", text)):
            raise ValueError("negative or stop instruction is not an executable task")
        started = time.monotonic()
        ids = self.tokenizer.encode(self.tokenizer.prompt(self.system, text))
        generated = self.model.greedy(self._tensor([ids]), self.tokenizer.eos_id)
        raw = self.tokenizer.decode(generated)
        generation_s = time.monotonic() - started
        return {"text": text, "task": validate_model_task(raw),
                "raw": raw, "generation_s": generation_s}
