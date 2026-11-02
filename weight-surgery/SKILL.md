---
name: weight-surgery
description: Diagnose a fault in a trained neural network by reading its weights and activations, then fix it with a small numerical edit (no gradient training) and prove the fix on held-out data. Use when a model has a specific, repeatable failure (looping, two labels confused, a class never predicted, a behaviour that should be absent) and you can load its weights and run forward passes. Also audits what changed between two checkpoints of one network. Not for adding an ability the model lacks.
---

# Weight surgery

A model is a set of matrices. Many faults are written in a few directions of a few of them. This skill
finds those directions with measurements, removes or moves them with a closed-form edit, and accepts
the edit only if held-out numbers say so.

The rule that decides everything: **an edit can uncover an ability that something hides; it cannot
create an ability the model does not have.** Step 3 tests which case you are in before you edit.

## Ground rules

1. Split the data before you start: a **search** set (look, measure, fit edits) and an **evaluation**
   set (look only at the final numbers, once per candidate edit). Never fit, choose or tune on the
   evaluation set. Split by source (file, user, document), not by row.
2. Count forward passes. Write the budget down before you start and report what you used.
3. Never overwrite the original weights. Write every edit to a new directory with a note of what changed.
4. Compare every edit against the cheap baselines: threshold or bias tuning, and more training data.
   An edit that ties a bias tweak is not worth its risk.
5. Report failures too. "The information is not in the activations" is a result.

## Step 1: name the failure and measure it

Write one sentence: *on inputs X the model does Y instead of Z*. Measure how often on the search set,
and keep 20 to 50 failing and passing examples. A fault that is a rate (e.g. 93% of answers hit the
token limit, or 37% of tokens of label A are predicted as B) is a good target. A vague "it is worse"
is not.

## Step 2: read the weights (cheap, no data)

Start with the matrices whose rows or columns have a meaning: the output head, the unembedding,
the output projections of attention and MLP blocks.

- `scripts/head_geometry.py MODEL_DIR [OTHER_DIR ...]` — for a classification head: one direction
  per class (mean of its tag rows minus the null row), the shared "is a span" component removed,
  then centred cosine between every pair. Classes near 0 use different directions; +0.5 and above use
  nearly the same one. Pass several checkpoints to see whether training moved them apart.
- Look for **initialisation fingerprints**: classes that were warm-started from the same row and that
  training did not separate. They show as high centred cosine exactly on the pairs that shared a source.
- Look at norms and biases per class (a class with a much lower bias is under-predicted) and at the
  singular values of the head (one huge value is the shared component; a long flat tail is normal).

Weight reading gives hypotheses only. Two similar rows can still be separated by the layers below.
In practice the pair with the most similar rows is often not the pair the model confuses: check the
real confusions in Step 3 before you edit anything.

## Step 3: read the activations (the decision gate)

Collect the vectors that feed the suspect matrix, on the search set, with the gold label (or a pass/fail
flag) and a source id per vector. Save them as a `.pt` file:
`{"hidden": float[N, d], "gold": long[N], "groups": list[str]}` (or `"flag"` instead of `"gold"`).

```python
out = model(**batch, output_hidden_states=True)
hidden = out.hidden_states[layer][0][keep]   # last layer for a head; any layer for a direction
```

Then ask whether the information is present:

- `scripts/probe_refit.py FEATURES.pt MODEL_DIR` refits only the head on half the groups (pulled
  towards the old head by an L2 term), scores the other half, swaps, and prints per-class precision,
  recall, F1 and top confusions for the old and the refit head.
  - Refit clearly better on the faulty classes → the information is there and the head blocks it.
    Edit the head (Step 5a).
  - Refit no better → the layers below do not separate the classes. Stop editing. The fix is data or
    training. Write that down.
- `scripts/find_direction.py FEATURES.pt` for a pass/fail flag: difference of class means, and the
  held-out-by-group AUC of the projection on it. Run it per layer; the layer where AUC jumps is where
  the behaviour is written. AUC near 0.5 everywhere → no single direction; do not ablate.

## Step 4: locate the writers

For a direction `v` found at layer `L`, score every matrix that writes into the residual stream before
`L` (attention output, MLP down projection) by `‖vᵀ W‖`, and every column `j` by `|vᵀ W[:, j]|`.
A fault written by a few matrices and a few columns is a good candidate for a small edit. A fault
spread thinly everywhere is not.

## Step 5: edit

Pick the smallest edit that matches the diagnosis. Write each to a new directory.

a. **Head refit with a pull** (information present, head blocks it):
   `probe_refit.py FEATURES.pt MODEL_DIR --write OUT_DIR` fits on all search groups and writes the
   new head. Pass `--only A,B` to move only the rows of the faulty classes and freeze the rest: a full
   refit tends to trade one class for another (it fits the search set's class mix), and a frozen row
   cannot get worse. Raise `--pull` until the change is small.
b. **Bias shift**: add a constant to one class's bias. The baseline every other edit must beat.
c. **Direction ablation** (a behaviour written along `v`):
   `scripts/ablate_direction.py MODEL_DIR DIRECTION.pt --tensor NAME [--tensor NAME] --out OUT_DIR`
   applies `W ← W − s · v (vᵀ W)` to the columns with the largest `|vᵀ W[:, j]|` (`--columns K`,
   default all) with strength `s` (`--scale`). Sweep `s` in {0.25, 0.5, 1} on the search set.

The edit's size is a result: report how many weights changed and what fraction of the model that is.

## Step 6: evaluate once

Run the original and each surviving edit on the evaluation set. Report:

- the target metric and its paired difference with a 95% interval from a bootstrap over source groups:
  `scripts/paired_bootstrap.py BASE.jsonl EDIT.jsonl` (one line per example, same order, with
  `group` and either `score` or `tp`/`fp`/`fn`);
- every other class or task, to catch damage the edit did elsewhere;
- the forward passes used and the number of weights changed.

Accept the edit only if the interval of the target excludes zero and nothing else gets worse beyond
its own interval. If several edits were tried, say how many: the evaluation set is now less clean.

## Step 7: write it up

One paragraph each: the failure and its rate; what the weights showed; what the activations showed
(present or absent); the edit, its size and the search-set sweep; the evaluation numbers with intervals;
what did not work.

## Audit mode: what changed between two checkpoints

Use it after a fine-tune, a continued run or a merge, or to tell two fine-tunes apart.

1. `scripts/diff_checkpoints.py A_DIR B_DIR` gives the relative change per tensor and per layer, and,
   for each matrix, the share of the change in its top singular direction. A share near 1 on a few
   matrices is a targeted change, worth reading. A small share everywhere is ordinary training drift.
   Tensors that were added or reshaped (a new head) are listed apart; read them with `head_geometry.py`.
2. Weights alone often cannot say what a change does. Run both checkpoints on a few texts you write
   for the question (an honest and a flattering answer, a date of a wedding and of a surgery), and
   compare the internal vectors or attention of the layers that changed most. A small behavioural probe
   on top of the weights is what decides the question.
3. Report: where the change is (layers, matrices), how concentrated it is, and what the probe showed.

## Credit

The approach follows Takehi, Teoh, Zhang and Khattab (2026), *Sparks of Weight-Space Programming:
LLMs can natively read & write neural network weights*,
https://rikiyat.github.io/blog/weight-space-programming/ — an agent removed a looping direction from
two MLP layers of Qwen3.5-0.8B (0.016% of the weights) and raised MATH-500 from 7.9% to 48.1%, and
could not improve a model that did not loop.
