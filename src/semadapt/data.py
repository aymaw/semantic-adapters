"""Template-family holdout, exact-text deduplication, and separate OOD examples."""

import hashlib
import json
import random

LABELS = ["track", "cancel", "refund", "address", "stock", "compare"]
TEMPLATES = {
    "track": [
        "where is my {item} order {num}",
        "track delivery of {item} {num}",
        "has my {item} shipment {num} arrived",
        "show shipping progress for {item} {num}",
        "when should order {num} containing {item} arrive",
        "can you locate the parcel {num} with my {item}",
        "i do not want to cancel {num} just track my {item}",
        "delivery estimate needed for my {item} purchase {num}",
    ],
    "cancel": [
        "cancel my {item} order {num}",
        "stop order {num} for {item}",
        "i want to cancel the {item} purchase {num}",
        "please void my {item} order {num}",
        "do not send the {item} i ordered {num}",
        "withdraw my purchase {num} of {item} before it ships",
        "not a refund i need order {num} for {item} cancelled",
        "please terminate the unshipped {item} purchase {num}",
    ],
    "refund": [
        "refund my {item} order {num}",
        "return the {item} from order {num}",
        "i need my money back for {item} {num}",
        "how can i send back {item} {num}",
        "the {item} arrived damaged give me a refund for {num}",
        "reimburse the cost of my delivered {item} {num}",
        "do not cancel anything return the delivered {item} {num}",
        "start a return for the {item} bought in {num}",
    ],
    "address": [
        "change the shipping address for {item} {num}",
        "update delivery location for {item} {num}",
        "send order {num} with {item} to a new address",
        "edit the destination of {item} {num}",
        "i moved reroute my {item} order {num}",
        "replace the postal destination for order {num} of {item}",
        "keep the {item} order {num} but change where it ships",
        "correct my street address on {item} shipment {num}",
    ],
    "stock": [
        "is {item} in stock {num}",
        "can i buy {item} today {num}",
        "do you have any {item} available {num}",
        "check inventory for {item} {num}",
        "will you restock the {item} soon {num}",
        "are there units of {item} ready to purchase {num}",
        "i have not ordered yet is {item} available {num}",
        "tell me whether the {item} is sold out {num}",
    ],
    "compare": [
        "compare {item} with {other} {num}",
        "how is {item} different from {other} {num}",
        "which is better {item} or {other} {num}",
        "show differences between {item} and {other} {num}",
        "help me choose between {item} and {other} {num}",
        "what advantages does {item} have over {other} {num}",
        "before buying contrast the {item} and {other} {num}",
        "explain the tradeoffs of {item} versus {other} {num}",
    ],
}
ITEMS = [
    "blue headphones",
    "travel charger",
    "red backpack",
    "desk lamp",
    "wireless mouse",
    "silver keyboard",
    "green speaker",
    "fitness watch",
    "coffee grinder",
    "phone stand",
]


def generate(seed=23, per_family=60):
    rng = random.Random(seed)
    splits = {k: [] for k in ("train", "validation", "test", "ood")}
    for label in LABELS:
        for family, template in enumerate(TEMPLATES[label]):
            split = "train" if family < 4 else "validation" if family == 4 else "test"
            for i in range(per_family):
                item, other = rng.sample(ITEMS, 2)
                text = template.format(item=item, other=other, num=f"x{family}{i:03d}")
                splits[split].append(
                    {"text": text, "label": LABELS.index(label), "family": f"{label}/{family}"}
                )
    ood_templates = [
        "write a poem about {item}",
        "what weather will we have near {item}",
        "translate {item} into french",
        "explain quantum physics using {item}",
        "who won the game about {item}",
        "diagnose a broken {item}",
    ]
    for template in ood_templates:
        for item in ITEMS:
            splits["ood"].append({"text": template.format(item=item), "label": -1, "family": "ood"})
    for rows in splits.values():
        rng.shuffle(rows)
    texts = [r["text"] for rows in splits.values() for r in rows]
    assert len(set(texts)) == len(texts)
    families = [set(r["family"] for r in splits[k]) for k in ("train", "validation", "test")]
    assert not (families[0] & families[1] or families[1] & families[2] or families[0] & families[2])
    return splits


def fingerprint(splits):
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
