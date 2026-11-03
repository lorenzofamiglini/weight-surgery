# Worked example: a news NER model on Wikipedia

`dslim/bert-base-NER` (BERT base, 108M weights) was trained on CoNLL-2003 news. On WikiANN English it
loses most of its accuracy. What can an edit by hand recover, following the post's rules: the weights
first, a budget of 50 forward passes (64 sentences each), no optimizer?

Search set: WikiANN validation. Evaluation set: WikiANN test (10,000 sentences), scored once per system
at the end. Groups for the bootstrap: blocks of 20 sentences (WikiANN has no documents). CPU only.

## Step 2: the weights (no forward pass)

- `head_geometry.py`: the head is a regular simplex. Every pair of classes sits at centred cosine
  −1/3, all biases are about 0, all norms are equal. No class is favoured or squeezed.
- `weight_votes.py --pair ORG,LOC` (and ORG,MISC, ORG,O): no neuron is aligned with any class
  direction beyond chance (at most 4.4 against about 4 for 3,072 random columns). Only the embeddings
  read clearly: *Academy*, *School*, *Army*, *College* point towards ORG.

The weights alone name no fault.

## Step 1 and 3: the failure (10 passes, 640 validation sentences)

Of 289 gold ORG names, 98 are found exactly. 54 (19%) come out as MISC, exactly or in part: songs
(*Bad Romance*), leagues (*Conference Premier*), shows. CoNLL calls these MISC; WikiANN calls them ORG.
Many other errors are span conventions: WikiANN puts *Carthay , Los Angeles* and
*Follin ( disambiguation )* in one span.

Diagnosis: the network finds these names; its MISC rows mean ORG in WikiANN's scheme. That is a label
mismatch, not a fault in the network.

## Step 4: the edit

`relabel.py` gives the two MISC rows of the head the ORG label. No number changes. Checked against
MISC off (a large negative bias on the MISC rows) on 640 other validation sentences (10 more passes,
20 of 50 used): micro-F1 0.480 → 0.503 (MISC off) and 0.513 (MISC labelled ORG).

## Step 5: evaluation (test split, once)

| System | Micro-F1 | Δ vs original [95% CI] | PER | ORG | LOC |
|---|---|---|---|---|---|
| Original | 0.479 | | 0.700 | 0.385 | 0.421 |
| MISC off (bias edit) | 0.501 | +0.023 [+0.021, +0.025] | 0.686 | 0.390 | 0.413 |
| **MISC rows labelled ORG** | **0.510** | **+0.032 [+0.029, +0.034]** | 0.700 | 0.410 | 0.421 |

The relabelled head beats MISC off by +0.009 [+0.006, +0.012], and PER and LOC do not move.

## What did not work, and the fitted control

The span conventions are not in the weights and no edit can teach them. A fitted control shows how
much is left: `probe_refit.py` refits the head with gradient steps on 10,000 labelled sentences and
reaches 0.588 (+0.109 [+0.102, +0.116]). That is fine-tuning of the head, not the post's method. An
earlier version of this page called it a weight edit with no training. That was wrong.

## Reproduce

```bash
uv sync --extra examples
cd examples/bert-ner-wikiann
uv run python -c 'from data import model; model()'
uv run python ../../weight-surgery/scripts/head_geometry.py model
uv run python ../../weight-surgery/scripts/weight_votes.py model --pair ORG,MISC --embedding bert.embeddings.word_embeddings.weight
uv run python relabel.py model relabelled
uv run python evaluate.py model test old.jsonl
uv run python evaluate.py model test nomisc.jsonl --drop MISC
uv run python evaluate.py relabelled test relabelled.jsonl
uv run python ../../weight-surgery/scripts/paired_bootstrap.py old.jsonl relabelled.jsonl
```

The fitted control: `collect.py validation features.pt`, then
`probe_refit.py features.pt model --write refit`, then `evaluate.py refit test refit.jsonl`.
