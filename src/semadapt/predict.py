"""Load the saved classifier, pinned base, and validation-only decision settings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .evaluate import probabilities
from .train import encode, infer


def load(checkpoint):
    directory = Path(checkpoint)
    config = json.loads((directory / "inference.json").read_text())
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    if config["mode"] == "lora":
        if config["smoke"]:
            base = AutoModelForSequenceClassification.from_pretrained(directory / "smoke-base")
        else:
            base = AutoModelForSequenceClassification.from_pretrained(
                config["base_model"],
                revision=config["revision"],
                num_labels=len(config["labels"]),
                attn_implementation="eager",
            )
        model = PeftModel.from_pretrained(base, directory)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(directory)
    return tokenizer, model, config


def predict(checkpoint, texts):
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Provide nonempty text inputs")
    tokenizer, model, config = load(checkpoint)
    logits = infer(model, encode(tokenizer, [{"text": t} for t in texts]))
    probs = probabilities(logits, config["temperature"])
    return [
        {
            "label": config["labels"][int(np.argmax(p))]
            if p.max() >= config["threshold"]
            else None,
            "confidence": float(p.max()),
            "abstained": bool(p.max() < config["threshold"]),
            "smoke_random_initialization": config["smoke"],
        }
        for p in probs
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.text), indent=2))
