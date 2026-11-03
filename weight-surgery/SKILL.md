---
name: weight-surgery
description: Diagnose a fault in a trained neural network by reading its weights, then fix it with a small edit written by reasoning (no optimizer, no gradient descent, a small budget of forward passes) and prove the fix on held-out data. Use when a model has a specific, repeatable failure (looping, two labels confused, a class never predicted, a behaviour that should be absent) and you can load its weights. Also audits what changed between two checkpoints of one network. Not for adding an ability the model lacks.
---

# Weight surgery

A model is a set of matrices. Many faults are written in a few directions of a few of them. This skill
reads the matrices to find those directions, writes an edit by hand, and accepts it only if held-out
numbers say so. It follows the method of the post credited at the end.

The rule that decides everything: **an edit can uncover an ability that something hides; it cannot
create an ability the model does not have.**

## Ground rules (from the post)

1. **The weights are the main object.** Read them first: rows, columns, products of matrices. Running
   the model is the scarce resource, not the default.
2. **A forward-pass budget, written down before you start** (the post used 0 to 50; one pass is one
   batch). Count every pass and report the count.
3. **No optimizer.** No gradient descent, no fitting to data. Every edit is a closed form you write and
   can explain: subtract a direction, scale or zero some neurons, change a bias, set a row by a formula.
4. **The evaluation set stays unseen** until the end: not for search, not for choosing a strength.
   Split by source (file, user, document), not by row.
5. Never overwrite the original weights. Write every edit to a new directory with a note.
6. Compare against the cheap baselines (a bias change, more training data) and report failures.
   "There is nothing to remove" is a result: the post's control model ended there.

## Step 1: name the failure

One sentence: *on inputs X the model does Y instead of Z*. Measure its rate with a few passes on the
search set. A rate (93% of answers hit the token limit; 37% of label A predicted as B) is a good target.

## Step 2: read the weights (no forward pass)

- `scripts/head_geometry.py MODEL_DIR [OTHER_DIR ...]` — for a classification head: one direction per
  class, the shared component removed, centred cosine between every pair, norms, biases, singular
  values. Look for **initialisation fingerprints** (classes warm-started from one row that training
  did not separate) and for rows that are exact copies.
- `scripts/weight_votes.py MODEL_DIR --pair A,B [--norm FINAL_NORM] [--embedding NAME]` — push the
  head's A-against-B direction back through every matrix that writes into the residual stream. Each
  neuron gets a vote for A or for B; each matrix gets the share of its weight along the direction;
  each vocabulary token gets a score. This names the neurons and layers to suspect.
- Products of matrices answer questions about circuits without running the model (the post found
  copying heads this way): for a head, `W_O W_V` says what it writes for what it reads, and
  `W_Q^T W_K` what it looks for.

Weight reading gives hypotheses only. The pair with the most similar rows is often not the pair the
model confuses.

## Step 3: test the hypothesis with a few passes

Spend the budget on inputs that split the hypothesis: minimal pairs you write (*the wedding on 3 May*
against *the surgery on 3 May*), and failing and passing examples from the search set.

- Check that the suspect neurons fire on the failures and not on the passes.
- `scripts/find_direction.py FEATURES.pt` gives the difference of the class means of failing and
  passing vectors (a measurement, not a fit) and its held-out AUC. Run it per layer: the layer where
  AUC jumps writes the behaviour. AUC near 0.5 everywhere means no single direction: do not ablate.

## Step 4: edit by hand

Write each edit to a new directory. Report how many weights changed and what fraction of the model
that is.

- **Direction removal** (a behaviour written along `v`, the post's fix for looping):
  `scripts/ablate_direction.py MODEL_DIR DIRECTION.pt --tensor NAME [--tensor NAME] --out OUT_DIR`
  applies `W ← W − s · v (vᵀ W)` to the K columns with the largest `|vᵀ W[:, j]|` (`--columns K`).
  Choose `s` and K by reasoning, then check a few values within the budget.
- **Neuron scaling**: scale the columns of the neurons Step 2 named and Step 3 confirmed.
- **Bias change**: add a constant to one class's bias. The baseline every other edit must beat.

## Step 5: evaluate once

Run the original and each surviving edit on the evaluation set. Report:

- the target metric and its paired difference with a 95% interval from a bootstrap over source groups:
  `scripts/paired_bootstrap.py BASE.jsonl EDIT.jsonl` (one line per example, same order, with
  `group` and either `score` or `tp`/`fp`/`fn`);
- every other class or task, to catch damage the edit did elsewhere;
- the forward passes used and the number of weights changed.

Accept the edit only if the interval of the target excludes zero and nothing else gets worse beyond
its own interval. Say how many edits were tried.

## Step 6: write it up

One paragraph each: the failure and its rate; what the weights showed; what the passes confirmed; the
edit and its size; the evaluation numbers with intervals; what did not work.

## Audit mode: what changed between two checkpoints

1. `scripts/diff_checkpoints.py A_DIR B_DIR` gives the relative change per tensor and per layer, and,
   for each matrix, the share of the change in its top singular direction. A share near 1 on a few
   matrices is a targeted change, worth reading. Added or reshaped tensors are listed apart.
2. Weights alone often cannot say what a change does: the post's agent failed on weights alone and
   succeeded with attention patterns on 16 texts it wrote. Do the same: a few texts written for the
   question, both checkpoints, the internals of the layers that changed most.
3. Report where the change is, how concentrated it is, and what the probe showed.

## A fitted control (outside the post's method)

`scripts/probe_refit.py FEATURES.pt MODEL_DIR` refits the head with gradient steps on half the
groups and scores the other half. It answers one question: is the information in the vectors that feed
the head? Use it as a control when Step 3 is unclear. Its `--write` output is a fine-tuned head, not
weight surgery: if you ship it, call it fine-tuning.

## Credit

The method follows Takehi, Teoh, Zhang and Khattab (2026), *Sparks of Weight-Space Programming:
LLMs can natively read & write neural network weights*,
https://rikiyat.github.io/blog/weight-space-programming/ — with a budget of 50 forward passes and no
gradient descent, an agent removed a looping direction from 64 output columns of two MLP layers of
Qwen3.5-0.8B (0.016% of the weights) and raised MATH-500 from 7.9% to 48.1%, and could not improve a
model that did not loop.
