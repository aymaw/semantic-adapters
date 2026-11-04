import torch

from semadapt.data import generate
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
