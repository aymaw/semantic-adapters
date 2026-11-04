# Error analysis

The 48 authored template families are divided before text generation. Models
cannot receive the test families during training. This is a stricter language
shift than a random utterance split, but the inventory remains small and synthetic.

The retained loss histories show decreasing training loss alongside stalled or
worsening held-out-family validation loss. Selecting a later training checkpoint
using test F1 would inflate the result; selection uses validation NLL instead.

`results/errors.json` lists the ten highest-confidence mistakes per method and
seed. Negated phrases such as “do not cancel ... just track” expose lexical cue
confusion. Cancellation/return vocabulary overlaps, and unseen paraphrases create
errors across multiple intent classes. Read the confusion matrices, rather than
attributing every failure to one cause. The reports do not establish a causal
diagnosis of the model's internal representations.

LoRA uses 8,966 trainable parameters including its saved classifier, versus
4,386,694 for full fine-tuning. This establishes a parameter-budget difference,
not superior accuracy. The 10% label experiment keeps test/validation fixed and
uses stratified training subsampling; its smaller optimization budget is a
confounder when comparing label efficiency.

Validation-only calibration/thresholds can fail under template shift. In
particular, ≥95% observed validation accuracy at the selected threshold does not
transfer as a guarantee to test or OOD inputs. Coverage, accepted accuracy, NLL,
ECE, and false acceptance are all reported, including null accepted accuracy when
no examples are accepted.

Next experiments should use independent template authors, a broader validation
inventory, stronger pretrained encoders, domain/slot holdouts, and human-reviewed
minimal pairs. Preserve this test set when comparing them. Do not claim the
current synthetic scores establish real-world NLU quality.

## Review revision

Order numbers no longer encode template partition. New reference and low-label
results use the same independent-identifier generator and unchanged learning
settings. The separate familiar-template report quantifies fitting on known
surface forms; it must not be pooled with the unseen-template test or described
as real-world generalization. Per-class and template-level reports expose where
aggregate F1 conceals systematic mistakes. Cluster intervals condition on a very
small authored inventory and should be interpreted cautiously.
