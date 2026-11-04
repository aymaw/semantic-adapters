import json

import torch

from semadapt.data import generate
from semadapt.predict import predict
from semadapt.train import encode, make_model


def test_lora_freezes_backbone_but_updates_adapter_and_head(tmp_path):
    data = generate(per_family=2)
    tokenizer, model = make_model("lora", True, tmp_path, data, 7)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    trainable = {k for k, v in model.named_parameters() if v.requires_grad}
    assert trainable
    assert all("lora_" in k or "modules_to_save" in k for k in trainable)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < sum(
        p.numel() for p in model.parameters()
    )
    inputs = encode(tokenizer, data["train"][:4])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss = model(**inputs, labels=torch.tensor([r["label"] for r in data["train"][:4]])).loss
    loss.backward()
    optimizer.step()
    changed = {k for k, v in model.named_parameters() if not torch.equal(before[k], v)}
    assert changed <= trainable
    assert any("lora_B" in k for k in changed)
    assert any("classifier" in k for k in changed)


def test_offline_saved_adapter_predicts_without_training_labels(tmp_path):
    data = generate(per_family=2)
    tokenizer, model = make_model("lora", True, tmp_path, data, 7)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "inference.json").write_text(
        json.dumps(
            {
                "labels": list(model.config.id2label.values()),
                "mode": "lora",
                "smoke": True,
                "temperature": 1.0,
                "threshold": 1.01,
            }
        )
    )
    result = predict(tmp_path, ["where is my blue headphones order x12345678"])
    assert result[0]["abstained"] and result[0]["label"] is None
    assert 0 < result[0]["confidence"] < 1
