"""Secondary closed-template check; never a substitute for the shifted test."""

import argparse
import json
from pathlib import Path

from .data import fingerprint, generate
from .evaluate import metrics, probabilities
from .predict import load
from .train import encode, infer


def evaluate(runs, out):
    # New slot combinations and IDs, but intentionally familiar template families.
    rows = generate(seed=1023, per_family=15)["train"]
    original = generate()
    assert not ({r["text"] for r in rows} & {r["text"] for v in original.values() for r in v})
    report = {
        "scope": "same-template synthetic evaluation; template memorization is possible",
        "data_sha256": fingerprint({"matched": rows}),
        "n": len(rows),
        "runs": [],
    }
    for settings in sorted(Path(runs).glob("*/inference.json")):
        tokenizer, model, config = load(settings.parent)
        if config["smoke"]:
            raise ValueError("Do not mix random smoke checkpoints with pretrained results")
        probs = probabilities(infer(model, encode(tokenizer, rows)), config["temperature"])
        report["runs"].append(
            {
                "checkpoint": settings.parent.name,
                "mode": config["mode"],
                **metrics(
                    probs,
                    [r["label"] for r in rows],
                    config["threshold"],
                    groups=[r["family"] for r in rows],
                ),
            }
        )
    if not report["runs"]:
        raise ValueError("No trained checkpoints found")
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print([(r["checkpoint"], round(r["macro_f1"], 3)) for r in report["runs"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/reference")
    parser.add_argument("--out", default="runs/matched-template.json")
    args = parser.parse_args()
    evaluate(args.runs, args.out)
