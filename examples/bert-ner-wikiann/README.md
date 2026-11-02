# Worked example: a news NER model on Wikipedia

`dslim/bert-base-NER` (BERT base, 108M weights) was trained on CoNLL-2003 news. On WikiANN English it
loses most of its accuracy. Can a weight edit, and no training of the network, recover some of it?

Search set: WikiANN validation (10,000 sentences). Evaluation set: WikiANN test (10,000 sentences),
scored once per system at the end. Groups for the bootstrap: blocks of 20 sentences (WikiANN has no
documents). CPU only.

## Step 1: the failure (search set)

Entity-level micro-F1 0.489. ORG recall is low (3,230 of 4,677 missed), LOC has 3,834 false positives,
and the model predicts 1,900 MISC entities on a dataset that has no MISC class.

## Step 3: the decision gate

`probe_refit.py` refits only the head on half the groups and scores the other half (word level):

| Class | Old head F1 | Refit head F1 |
|---|---|---|
| ORG | 0.496 | 0.731 |
| LOC | 0.642 | 0.759 |
| PER | 0.804 | 0.846 |
| O | 0.886 | 0.931 |

The vectors that feed the head already separate the entities much better than the head does: the
information is present, the head blocks it.

## Step 5: how small can the edit be?

Pull sweep (word-level ORG F1 on held-out halves): head change 2% → 0.506, 18% → 0.533,
46% → 0.635, 125% → 0.731. Here the gain needs a new head, not a small cut. The new head is still
only 6,921 numbers, 0.006% of the network.

## Step 6: evaluation (test split, once)

| System | Micro-F1 | Δ vs old [95% CI] | PER | ORG | LOC |
|---|---|---|---|---|---|
| Old model | 0.479 | | 0.700 | 0.385 | 0.421 |
| Bias baseline: MISC off | 0.501 | +0.023 [+0.021, +0.025] | 0.686 | 0.390 | 0.413 |
| Head refit | **0.588** | **+0.109 [+0.102, +0.116]** | 0.745 | 0.489 | 0.541 |

The edit beats the bias baseline by a wide margin and improves every class. What is left of the gap is
not in the head: it needs the layers below, that is, training.

## What did not work

Small edits. With the head held within 2% of the original, nothing changed. Unlike the looping model
in the paper, this fault is not one direction; it is the whole mapping from vectors to classes.

## Reproduce

```bash
uv sync --extra examples
cd examples/bert-ner-wikiann
uv run python evaluate.py "$(uv run python -c 'from data import model; print(model())')" validation old-val.jsonl
uv run python collect.py validation features.pt
uv run python ../../weight-surgery/scripts/probe_refit.py features.pt model
uv run python ../../weight-surgery/scripts/probe_refit.py features.pt model --write refit
uv run python evaluate.py model test old.jsonl
uv run python evaluate.py model test nomisc.jsonl --drop MISC
uv run python evaluate.py refit test refit.jsonl
uv run python ../../weight-surgery/scripts/paired_bootstrap.py old.jsonl refit.jsonl
```

About 30 minutes on a laptop CPU.
