import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from .data import LABELS, fingerprint, generate
from .evaluate import fit_temperature, metrics, probabilities, select_threshold

MODEL = "google/bert_uncased_L-2_H-128_A-2"
REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"


def make_model(mode, smoke, out, data, seed):
    torch.manual_seed(seed)
    if smoke:
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(
            {w for row in data["train"] for w in row["text"].lower().split()}
        )
        (out / "vocab.txt").write_text("\n".join(vocab) + "\n")
        tokenizer = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=True)
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=len(vocab),
                hidden_size=48,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=96,
                num_labels=len(LABELS),
                max_position_embeddings=128,
            )
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, revision=REVISION, num_labels=len(LABELS), attn_implementation="eager"
        )
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {label: i for i, label in enumerate(LABELS)}
    if mode == "lora":
        if smoke:
            model.save_pretrained(out / "smoke-base")
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
                revision=None if smoke else REVISION,
            ),
        )
    elif mode == "head":
        for param in model.bert.parameters():
            param.requires_grad = False
    elif mode != "full":
        raise ValueError(mode)
    return tokenizer, model


def encode(tokenizer, rows):
    return tokenizer(
        [r["text"] for r in rows],
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )


def infer(model, inputs):
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                model(**{k: v[i : i + 64] for k, v in inputs.items()}).logits
                for i in range(0, len(inputs["input_ids"]), 64)
            ]
        ).numpy()


def evaluate_all(logits, data):
    val_labels = np.array([r["label"] for r in data["validation"]])
    temperature = fit_temperature(logits["validation"], val_labels)
    threshold = select_threshold(probabilities(logits["validation"], temperature), val_labels)
    test_labels = np.array([r["label"] for r in data["test"]])
    families = [r["family"] for r in data["test"]]
    return {
        "temperature": temperature,
        "validation_threshold": threshold,
        "uncalibrated_test": metrics(probabilities(logits["test"]), test_labels, groups=families),
        "calibrated_test": metrics(
            probabilities(logits["test"], temperature), test_labels, threshold, groups=families
        ),
        "ood_n": len(data["ood"]),
        "ood_false_accept_rate": float(
            (probabilities(logits["ood"], temperature).max(1) >= threshold).mean()
        ),
    }


def run(
    out="runs/reference",
    smoke=False,
    epochs=8,
    seeds=(7, 19, 31),
    modes=("head", "lora", "full"),
    train_fraction=1.0,
):
    out = Path(out)
    if epochs < 1 or not seeds or not modes:
        raise ValueError("Use positive epochs and nonempty seed/mode lists")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = generate(per_family=4 if smoke else 60)
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if train_fraction < 1:
        # Stratified, fixed-seed label budget; keep validation/test unchanged.
        rng = np.random.default_rng(101)
        selected = []
        for label in range(len(LABELS)):
            rows = [r for r in data["train"] if r["label"] == label]
            idx = rng.permutation(len(rows))[: max(1, int(len(rows) * train_fraction))]
            selected.extend(rows[i] for i in idx)
        data["train"] = selected
    (out / "dataset.json").write_text(json.dumps(data, indent=2))
    report = {
        "synthetic_only": True,
        "protocol": "template-holdout-v2-independent-order-identifiers",
        "smoke_random_initialization": smoke,
        "base_model": None if smoke else MODEL,
        "revision": None if smoke else REVISION,
        "data_sha256": fingerprint(data),
        "split_counts": {k: len(v) for k, v in data.items()},
        "epochs": epochs,
        "train_fraction": train_fraction,
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "runs": [],
    }
    baseline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=300, C=4)
    )
    baseline.fit([r["text"] for r in data["train"]], [r["label"] for r in data["train"]])
    logits = {
        k: np.log(baseline.predict_proba([r["text"] for r in data[k]]).clip(1e-12))
        for k in ("validation", "test", "ood")
    }
    report["tfidf_baseline"] = evaluate_all(logits, data)
    for seed in seeds:
        for mode in modes:
            started = time.perf_counter()
            destination = out / f"{mode}-{seed}"
            destination.mkdir(exist_ok=True)
            tokenizer, model = make_model(mode, smoke, destination, data, seed)
            inputs = {k: encode(tokenizer, rows) for k, rows in data.items()}
            labels = {k: torch.tensor([r["label"] for r in rows]) for k, rows in data.items()}
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=5e-4 if mode != "full" else 5e-5,
                weight_decay=0.01,
            )
            best, checkpoint, history = float("inf"), None, []
            for epoch in range(epochs):
                model.train()
                loss_sum = 0.0
                for idx in torch.randperm(len(data["train"])).split(32):
                    output = model(
                        **{k: v[idx] for k, v in inputs["train"].items()},
                        labels=labels["train"][idx],
                    )
                    optimizer.zero_grad()
                    output.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    loss_sum += output.loss.item() * len(idx)
                val_logits = infer(model, inputs["validation"])
                val_loss = torch.nn.functional.cross_entropy(
                    torch.tensor(val_logits), labels["validation"]
                ).item()
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_nll": loss_sum / len(data["train"]),
                        "val_nll": val_loss,
                    }
                )
                if val_loss < best:
                    best, checkpoint = val_loss, copy.deepcopy(model.state_dict())
            model.load_state_dict(checkpoint)
            logits = {k: infer(model, inputs[k]) for k in ("validation", "test", "ood")}
            evaluation = evaluate_all(logits, data)
            calibrated = probabilities(logits["test"], evaluation["temperature"])
            errors = [
                {
                    "text": row["text"],
                    "family": row["family"],
                    "gold": LABELS[row["label"]],
                    "predicted": LABELS[int(p.argmax())],
                    "confidence": float(p.max()),
                }
                for row, p in zip(data["test"], calibrated)
                if p.argmax() != row["label"]
            ]
            errors.sort(key=lambda x: -x["confidence"])
            (destination / "errors.json").write_text(json.dumps(errors, indent=2))
            model.save_pretrained(destination)
            tokenizer.save_pretrained(destination)
            (destination / "inference.json").write_text(
                json.dumps(
                    {
                        "labels": LABELS,
                        "mode": mode,
                        "smoke": smoke,
                        "base_model": None if smoke else MODEL,
                        "revision": None if smoke else REVISION,
                        "temperature": evaluation["temperature"],
                        "threshold": evaluation["validation_threshold"],
                    },
                    indent=2,
                )
                + "\n"
            )
            # Verify adapter serialization and reload, including the trained classification head.
            if mode == "lora":
                _, fresh = make_model("full", smoke, destination, data, seed)
                restored = PeftModel.from_pretrained(fresh, destination)
            else:
                restored = BertForSequenceClassification.from_pretrained(destination)
            roundtrip = float(np.max(np.abs(infer(restored, inputs["test"]) - logits["test"])))
            if roundtrip > 1e-5:
                raise RuntimeError(f"Checkpoint reload changed predictions: {roundtrip}")
            result = {
                "seed": seed,
                "mode": mode,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "best_val_nll": best,
                "history": history,
                "roundtrip_max_abs_error": roundtrip,
                "seconds": time.perf_counter() - started,
                **evaluation,
            }
            report["runs"].append(result)
            np.savez_compressed(destination / "logits.npz", **logits)
            print(
                f"{mode}-{seed}: F1={result['calibrated_test']['macro_f1']:.3f}, trainable={trainable}",
                flush=True,
            )
            (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/reference")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31])
    p.add_argument("--modes", nargs="+", default=["head", "lora", "full"])
    p.add_argument("--train-fraction", type=float, default=1.0)
    a = p.parse_args()
    run(
        a.out,
        a.smoke,
        a.epochs,
        [7] if a.smoke else a.seeds,
        ["lora"] if a.smoke else a.modes,
        a.train_fraction,
    )
