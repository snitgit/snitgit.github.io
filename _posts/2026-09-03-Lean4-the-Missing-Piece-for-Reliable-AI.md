---
layout: post
title:  "Lean 4 Might Be the Missing Piece: Optimise, Then Prove"
date:   2026-09-03 09:00:00 +0700
categories: AI FormalMethods
tags: [lean4, formal-verification, reliable-ai, optimization, 6g, thailand]
---

Modern AI is very good at finding an answer and very bad at telling you when that answer is wrong. This post is an argument for pairing every optimiser with a proof — and for why Lean 4 is a good place to do it.

## The gap nobody wants to own

An optimiser — gradient descent, an evolutionary search, a reinforcement-learning policy, a large model doing in-context search — is a machine for producing *candidates*. Give it an objective and it will hand you something that scores well on that objective. What it will not hand you is a guarantee. It cannot tell you that the design is stable under every input in the operating range, that the bound holds in the worst case and not just the sampled cases, or that the property you actually care about survives the corner you forgot to test.

We paper over this with benchmarks. A benchmark score is a measurement, and every measurement can be gamed: overfit the test set, tune to the metric, pick the seed that worked. An LLM acting as a judge has the same problem one level up — it is another model with its own blind spots. None of these produce something a sceptic cannot argue with.

A machine-checked proof does. That is the whole point of it.

## What each half gives you

| An optimiser gives you | A proof gives you |
|---|---|
| A candidate that scores well **on what you measured** | A guarantee about **every case in the stated domain** |
| Fast iteration, no need to understand the search space | A spec you were forced to write down explicitly |
| A number that can be gamed, overfit, or cherry-picked | A pass/fail a referee — human or machine — cannot talk their way around |
| "It worked on the benchmark" | "It is impossible for this to fail, given these assumptions" |
| Silence when an assumption breaks | A specific pointer to the assumption that was load-bearing |

Neither half is sufficient. An optimiser with no proof is a confident guess. A proof with no optimiser is a blank page — you still need something to generate the candidate worth checking. The useful loop is both, in sequence:

```
        +----------------------------------------------+
        |                                              |
        v                                              |
   [ OPTIMISE ] --> candidate --> [ FORMALISE ] --> [ CHECK ]
   search / ML /     design,       state the         Lean 4
   tuning            params        guarantee         verifies
                                                        |
                          +-----------------------------+
                          |                             |
                    proof FAILS                   proof PASSES
                          |                             |
                          v                             v
              tells you which assumption      a certified result:
              was load-bearing --> tighten    stays true forever,
              the spec or re-run the search    re-checkable by anyone
```

The failure branch is the one people underestimate. When a formal proof does not close, it does not just say "no" — it leaves you staring at the exact hypothesis you could not discharge. That is often more valuable than the proof succeeding, because it is where the real engineering assumption was hiding.

## Why Lean 4 specifically

A proof assistant is a program that checks a proof step by step and refuses to accept one that does not follow. Lean 4 is one of several — Coq, Isabelle and Agda are others — and any of them supports the argument here. Lean 4 is a reasonable default for new work in this space for a few practical reasons:

- **It is a checker, and its output cannot be gamed.** The kernel either accepts the proof term or it does not. There is no partial credit, no seed to pick, no metric to tune. For a field drowning in benchmark inflation, that property alone is worth a lot.
- **It is also a real programming language.** The same system compiles and runs code, so the object you verify and the object you execute can be closer together than in a pen-and-paper argument.
- **It has an active mathematical library** built by a large open community, which means many standard results are already formalised and reusable rather than something you must rebuild.
- **The AI-for-theorem-proving effort is converging on it.** A growing share of research on machine-assisted proof search targets Lean, so tooling that helps a model *find* proofs is landing here first.

That last point closes the loop in a second sense: the optimiser that proposes the design and the search that helps discharge the proof can both be learned systems, with the kernel as the one component that is not.

## A concrete anchor, without the details

To keep this from being abstract: I have been working on a formal proof, in Lean 4, about the behaviour of a phase-locked loop — a control component that shows up everywhere in radio, and that matters for 6G wireless in particular.

I am not going to describe the result here. What is relevant is the *shape* of the work, because that shape is reusable:

1. A design process — partly search, partly hand-tuning — produces a set of parameters that performs well in simulation.
2. The guarantee you actually want ("this stays well-behaved across the whole operating range") is not something simulation can establish. It has to be stated as a formal claim and proved.
3. Writing that claim down forces every implicit assumption into the open — operating ranges, noise models, approximations that were fine "in practice."
4. The proof either closes, and you have a guarantee that does not depend on which cases you happened to simulate, or it does not, and you now know precisely which assumption your design was silently relying on.

None of that is specific to phase-locked loops or to wireless. Replace step 1 with "a neural network architecture search," or "a compiler optimisation pass," or "a scheduling policy," and steps 2 through 4 are unchanged. The workflow transfers; the domain is just where you happen to be standing.

## Why this is a good bet for Thai science and engineering

Here is the asymmetry that makes this worth writing about.

Training frontier models is **capital-bound**: it needs clusters, energy budgets and data pipelines that only a handful of organisations can afford, and the results are mostly closed. Trying to compete there from a university lab is a losing position.

Formal verification is **skill-bound**: it needs a person who understands a problem well enough to state its guarantee precisely, and a laptop. The output is a checked theorem that goes into a shared library and *stays checked* — forever, re-verifiable by anyone, no trust in the author required. The barrier is entirely education, and education is something a research community can build deliberately.

That is a field where a small, well-trained group can produce work that sits permanently in the international record, at a fraction of the cost of trying to out-scale anyone. For Thai engineering departments with strong traditions in control, communications, power systems and signal processing — all fields full of results that have never been formalised — the raw material is already on the shelves.

Three concrete first steps, in order of effort:

1. **Learn the tool.** Work through the official Lean documentation and its interactive tutorial (`lean-lang.org`). A motivated engineering student can be writing real proofs within a few weeks.
2. **Formalise one result you already know.** Take a theorem from your own coursework or research — a stability criterion, a bound, a convergence result — and state and prove it in Lean. This is how the skill actually transfers into your field.
3. **Put an optimiser next to a checker.** Take a design task you already solve numerically, write down the guarantee you wish you had, and try to prove it. Whether it closes or not, you will learn something about the design you did not know.

The reliable-AI problem — systems whose behaviour we can actually depend on — is not going to be solved by scale alone. It needs a layer that says *proved*, not *scored*. That layer is open, it is cheap to enter, and there is room in it for a lot more people.
