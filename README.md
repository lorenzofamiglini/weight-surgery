# weight-surgery

An open agent skill for finding a fault inside a trained neural network by reading its weights and
activations, fixing it with a small closed-form edit (no gradient training), and proving the fix on
held-out data.

It is written in the [Agent Skills](https://agentskills.io) format: a folder with a `SKILL.md` and
scripts. Any agent that loads skills can use it, and a person can follow `SKILL.md` by hand.

## Why

Takehi, Teoh, Zhang and Khattab (MIT CSAIL) showed that a frontier model, given a Python kernel with
the weights and a budget of forward passes, can find the direction that makes Qwen3.5-0.8B loop and
remove it from two MLP layers. The edit changed 0.016% of the weights and raised MATH-500 from 7.9%
to 48.1%. On a model that did not loop, the same search found nothing to fix:

> Editing weights could remove an obstacle that hid an ability, but it could not add an ability the
> model lacked.

This skill turns that into a repeatable procedure with a decision gate for exactly that question.

- Post: [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (2026)

## The procedure

1. Name the failure and measure its rate on a **search** set. Keep an **evaluation** set you never fit on.
2. Read the weights: class directions of the head, initialisation fingerprints, norms, biases.
3. Read the activations: is the information present? (head refit probe, direction AUC per layer)
   If not, stop: the fix is data or training.
4. Locate the matrices and columns that write the direction.
5. Edit: head refit with an L2 pull, bias shift, or direction ablation on the top-K columns.
6. Evaluate once, with a paired bootstrap over source groups, and check everything else for damage.
7. Write up what worked and what did not.

## Scripts

All take a Hugging Face style model directory (`*.safetensors`, optional `config.json` with `id2label`).

| Script | What it does |
|---|---|
| `head_geometry.py` | Centred cosine between class directions of a classification head, across checkpoints |
| `probe_refit.py` | Refit only the head on half the groups, score the other half; optionally write the refit head |
| `find_direction.py` | Difference-of-means direction for a pass/fail flag, with held-out AUC |
| `ablate_direction.py` | Remove a direction from what a matrix writes or reads, on the top-K columns |
| `paired_bootstrap.py` | Paired difference with a 95% interval from a bootstrap over groups |

Requirements: Python 3.10+, `torch`, `safetensors`.

## Install

Copy the `weight-surgery/` folder into your agent's skills directory (for Claude Code:
`~/.claude/skills/`), or give your agent the path to `SKILL.md`.

## License

MIT
