# weight-surgery

An open agent skill for finding a fault inside a trained neural network by reading its weights, fixing
it with a small edit written by reasoning (no optimizer, no gradient descent, a small budget of forward
passes), and proving the fix on held-out data.

It is written in the [Agent Skills](https://agentskills.io) format: a folder with a `SKILL.md` and
scripts. Any agent that loads skills can use it, and a person can follow `SKILL.md` by hand.

## Why

Takehi, Teoh, Zhang and Khattab (MIT CSAIL) showed that a frontier model, given a Python kernel with
the weights and a budget of forward passes, can find the direction that makes Qwen3.5-0.8B loop and
remove it from two MLP layers. The edit changed 0.016% of the weights and raised MATH-500 from 7.9%
to 48.1%. On a model that did not loop, the same search found nothing to fix:

> Editing weights could remove an obstacle that hid an ability, but it could not add an ability the
> model lacked.

This skill turns that into a repeatable procedure with the post's rules: the weights first, a written
forward-pass budget, and edits written by hand. It covers the post's *edit* experiments (a stalled
physics policy, the looping model and its control) and, as an audit mode, its *read* experiments (what a network learned, which of two fine-tunes is
sycophantic). The post's lesson from the sycophancy case holds here too: weights alone give hypotheses;
a small probe on real inputs decides.

- Post: [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (2026)

## Result on a public model

[`dslim/bert-base-NER`](examples/bert-ner-wikiann/), a news NER model, used on Wikipedia text (WikiANN
test, scored once):

| System | Method | Micro-F1 | Δ [95% CI] |
|---|---|---|---|
| Original | | 0.479 | |
| MISC off (one bias per MISC tag) | weight surgery: an edit by hand | 0.501 | +0.023 [+0.021, +0.025] |
| Head refit on 10,000 sentences | fitted control: gradient steps on the head | 0.588 | +0.109 [+0.102, +0.116] |

The hand edit is small and certain. The fitted head shows that the information is in the network and
how far a fitted head goes; it is fine-tuning of the head, not the post's method. An earlier version of
this README called it a weight edit with no training. That was wrong: the head was trained.

## The procedure

1. Name the failure and measure its rate on a **search** set. Keep an **evaluation** set you never see.
2. Read the weights: class directions of the head, initialisation fingerprints, the neurons that vote
   for one class against another, products of matrices. No forward pass.
3. Test the hypothesis with a few forward passes, on inputs you write and on search examples.
4. Edit by hand: remove a direction, scale neurons, change a bias. No optimizer.
5. Evaluate once, with a paired bootstrap over source groups, and check everything else for damage.
6. Write up what worked and what did not, with the passes used and the weights changed.

## Scripts

All take a Hugging Face style model directory (`*.safetensors`, optional `config.json` with `id2label`).

| Script | What it does |
|---|---|
| `head_geometry.py` | Centred cosine between class directions of a classification head, across checkpoints |
| `weight_votes.py` | Weights only: which neurons, layers and tokens vote for class A against class B |
| `find_direction.py` | Difference-of-means direction for a pass/fail flag, with held-out AUC |
| `ablate_direction.py` | Remove a direction from what a matrix writes or reads, on the top-K columns |
| `paired_bootstrap.py` | Paired difference with a 95% interval from a bootstrap over groups |
| `diff_checkpoints.py` | Audit: what changed between two checkpoints, per tensor and per layer, and how concentrated |
| `probe_refit.py` | Fitted control, outside the post's method: is the information in the vectors that feed the head? |

Requirements: Python 3.10+, `torch`, `safetensors`.

## Install

Copy the `weight-surgery/` folder into your agent's skills directory (for Claude Code:
`~/.claude/skills/`), or give your agent the path to `SKILL.md`.

## License

MIT
