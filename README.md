# EXOSFEAR

**Called EXOSFEAR because Skynet is taken.**

A repo with two moods:

- **`exosfear.py`** — a dangerous, compute-hungry, graph-shaped LLM training experiment
- **`exosfearminilab.py`** — a cleaner mini-lab for graph-law induction and benchmarked equation-family prediction

One file is a storm cellar with wires hanging out of the walls.
The other is a laboratory bench with labeled glassware.
They belong together.

---

## Read this first

### Caution: high risk, high compute, review deeply before running

**`exosfear.py` is not a casual script.**

It may:

- drive CPU or GPU usage hard
- consume a large amount of RAM or VRAM
- read a lot of local text if you aim it at broad directories
- run long enough to make a machine sluggish or unresponsive
- **lock up your computer** if launched carelessly on the wrong hardware or with the wrong settings

Treat it as **experimental high-risk code**.
Do **not** run it blindly.
Do **not** point it at a machine full of important work and then wander off.
Do **not** assume the defaults are gentle just because they are defaults.

Before trying `exosfear.py`, it should be **deeply reviewed**:

- review the ingestion paths
- review the training loops
- review memory use
- review checkpoint behavior
- review the directories it can touch and read
- review whether your hardware can tolerate sustained load

If you want the safer starting point, begin with **`exosfearminilab.py`**.
That file exists precisely because sometimes the correct first move is not to summon the machine, but to build the benchmark.

---

## What this repo is

EXOSFEAR is a two-headed project about machine structure.

Not just language.
Not just output.
Not just vibes.

It asks two related questions:

1. **Can a graph of small neural subsystems learn when to consult one another?**
2. **Can a system infer the hidden law behind a graph from its shape, statistics, and fragments?**

Those questions became two files.

### `exosfear.py`
A graph-organized neural training experiment.
Small transformer-like nodes. Different context windows. Learned routing. Gated message passing. Local and global training. Collaborative generation. Synthetic replay. Checkpoints. Ambition in a trench coat.

### `exosfearminilab.py`
A more honest instrument.
A benchmark harness for graph equation-family induction: generate synthetic graph families, expose multiple structural lenses, evaluate guesses, surface stage-0, midstage, and final metrics, and force the whole thing to stand on numbers rather than atmospherics.

If `exosfear.py` is the creature, `exosfearminilab.py` is the cage, ruler, and blood panel.

---

## Repo philosophy

The point of this repo is not to declare victory.
The point is to build things that can **fail interestingly and measurably**.

That means keeping both halves:

- the unruly training organism
- the clean benchmark lab

Without the lab, the organism becomes mythology.
Without the organism, the lab becomes homework.

Together, they become a research program.

---

## File guide

## `exosfear.py`

This is the wilder file.
It is the one that tries to grow a graph-shaped training system out of text, consultation, and recurrence.

At a high level it:

- ingests local text
- builds multiple neural nodes
- gives them different context spans and depths
- learns routing over which neighbor to consult
- injects gated messages between nodes
- alternates local and graph-level training
- generates collaborative output
- feeds some of that output back into the corpus
- checkpoints the graph over generations

It is not a product.
It is not a stable research artifact.
It is not something to trust because it prints something uncanny.

It is an experiment in whether structure can create the beginnings of distributed consultation.

### Why `exosfear.py` is dangerous

Because it is a real training script, not a postcard.

It has exactly the sort of properties that make experimental code exciting and hazardous:

- broad ingestion behavior
- long loops
- recursive growth tendencies
- synthetic replay
- multiple training phases
- potentially heavy tensor allocation
- enough moving parts to surprise you

This is the file most likely to make a laptop fan sound philosophical.

---

## `exosfearminilab.py`

This is the grounded half.
It does not pretend to be a mind.
It asks a cleaner question:

> Given the shape of a graph, can a system infer the family of law that generated it?

It creates synthetic graph worlds from families such as:

- Erdős–Rényi
- Barabási–Albert
- Watts–Strogatz
- stochastic block models
- random geometric graphs

Then it exposes multiple views of those graphs through different lenses, such as:

- local structure
- global structure
- spectral structure
- edge glimpses

And then it asks for a two-line response:

```text
LAW family=<family>; <param>=<value>
SELF confidence=<0..1>; alt_family=<family>; why=<brief reason>
```

It can surface benchmark snapshots at:

- **Stage 0** — before anything impressive happens
- **Midstage** — after tuning or validation-time calibration
- **Completed** — final held-out results

That matters because it turns the repo from a vibe machine into something that can actually say:

- here is the baseline
- here is the improvement
- here is what the system still gets wrong

---

## Why the pairing matters

A lot of ambitious AI repos make one of two mistakes.

### Mistake 1: pure theater
A dramatic architecture with no benchmark, no discipline, and no way to tell whether it learned something real.

### Mistake 2: pure hygiene
A perfectly neat benchmark that never risks building anything strange enough to discover a new failure mode.

EXOSFEAR tries not to choose between them.

- `exosfear.py` is the dangerous speculation
- `exosfearminilab.py` is the instrument panel

That contrast is the repo.

---

## Suggested order of operations

If you are new here:

1. **Read the code before running anything**
2. **Start with `exosfearminilab.py`**
3. Generate a benchmark and inspect Stage 0
4. Run the baseline and read the completed report
5. Only then consider whether `exosfear.py` deserves a controlled run on your hardware

This is not false modesty.
It is survival.

---

## What success looks like

For `exosfearminilab.py`, success looks like:

- reproducible benchmark generation
- sane stage reports
- meaningful held-out accuracy
- visible failure cases
- a solver that can be beaten by something better

For `exosfear.py`, success looks like something humbler than science fiction:

- evidence of specialization that is not fake
- routing that matters rather than decorates
- cleaner train/inference alignment
- measurable benefit over simpler baselines
- fewer theatrical claims and more controlled results

---

## What this repo is not claiming

Let us be adults for a moment.

This repo does **not** claim:

- AGI
- consciousness
- a true knowledge graph
- autonomous truth-seeking
- safe self-improvement
- reliable introspection just because a script prints a `SELF` line

It claims something narrower and, in a way, more interesting:

That it may be possible to build systems where **structure matters** — where separate lenses, separate nodes, and explicit consultation create behaviors worth measuring.

That is enough.
That is already a lot.

---

## Why the name still works

**EXOSFEAR** sounds like:

- a cybernetic shell
- a fear that lives outside the organism
- an industrial accident with a grant proposal
- a 1990s AI thriller with a VHS cover showing blue lightning and a red eye

All acceptable.

But the plain reason remains:

**It is called EXOSFEAR because Skynet is taken.**

---

## Practical warning, one more time

If you clone this repo and run `exosfear.py` without review because the name is funny, you are misunderstanding the joke.

The joke is the name.
The compute load is not a joke.
The lock-up risk is not a joke.
The need for deep review is not a joke.

Start with `exosfearminilab.py`.
Earn the right to launch `exosfear.py`.

---

## Final note

This repo contains both a beast and a measuring stick.
That is intentional.

One half asks whether a graph of little learners can start to consult itself.
The other asks whether any of that instinct can survive contact with a benchmark.

That is wild.
That is grounded.
That is the point.
