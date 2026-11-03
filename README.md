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

## What this is, and what it is not

The edits here are not new techniques. Removing a direction from what a matrix writes is directional
ablation by weight orthogonalisation ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)), a
form of steering ([Turner et al., 2023](https://arxiv.org/abs/2308.10248)). The post's loop fix is
this kind of edit, and its attempt to raise maths ability on a model that did not loop found nothing.
Rank-one edits to stored facts ([ROME](https://rome.baulab.info/),
[MEMIT](https://memit.baulab.info/)), task arithmetic
([Ilharco et al., 2023](https://arxiv.org/abs/2212.04089)) and learning on weights as data are older,
active fields. Much of the post's evidence comes from very small networks, which a person can also
read and edit by hand.

What the skill adds is discipline, not power: weights read before inputs, a counted pass budget, no
optimizer, an evaluation set kept unseen, a paired interval, and a fitted control next to every hand
edit. In our runs the control wins: on the NER model above it gives +0.109 against +0.032 for the
hand edit. Use the skill to find and explain a specific fault. Do not expect it to beat training.

## Result on a public model

[`dslim/bert-base-NER`](examples/bert-ner-wikiann/), a news NER model, used on Wikipedia text (WikiANN
test, scored once):

| System | Method | Micro-F1 | Δ [95% CI] |
|---|---|---|---|
| Original | | 0.479 | |
| MISC off | bias edit by hand | 0.501 | +0.023 [+0.021, +0.025] |
| **MISC rows labelled ORG** | **edit by hand, 20 forward passes, no number changed** | **0.510** | **+0.032 [+0.029, +0.034]** |
| Head refit on 10,000 sentences | fitted control: gradient steps on the head | 0.588 | +0.109 [+0.102, +0.116] |

The weights alone named no fault (a regular-simplex head, no neuron above chance). Twenty passes
showed a label mismatch: WikiANN calls songs, leagues and shows ORG, the model calls them MISC. The hand
edit fixes that and nothing else. The rest of the gap is span conventions, which need training: the
fitted head shows how far that goes, and it is fine-tuning, not the post's method. An earlier version
of this README called the fitted head a weight edit with no training. That was wrong.

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
