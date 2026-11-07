# Weight Space Programming: Survey of Formal Work

Survey date: 2026-11-07. Status: complete for the scope below. Items marked **not verified** are listed in section 11.

## 1. Scope and method

**Question.** What formal work already exists on building transformers and CNNs by *writing* their weights from a program or spec rather than training? Which of it can wsp reuse, given wsp's pipeline: NL intent → LLM → spec → weight compiler → PyTorch model on MPS → verification?

**Method.** Primary sources only:
- arXiv abstracts and PDFs, and proceedings pages;
- official GitHub repos, read through cloned source, READMEs, `git log`, and the GitHub REST API for push dates, archive flags and licenses.
Secondary sources (blogs, aggregators) appear only where noted, and only for history or leads. Repo facts are as of 2026-11-07.

**Summary of findings.**
1. The transformer side is well covered: RASP (language) → Tracr (compiler) → ALTA (a more capable compiler).
2. Tracr is archived and JAX/Haiku-based. **TransformerLens now ships a maintained Tracr → PyTorch converter**, so the practical path to a PyTorch model already exists.
3. On the CNN side there is **no Tracr equivalent**. There are only analytic constructions: scattering wavelets (Kymatio), CA-as-CNN, fixed Sobel perception.
4. On the LLM side, the only primary measurement (TracrBench) shows GPT-4-turbo writes correct RASP only 57/101 times. A validate-and-repair loop is therefore essential.
5. Learned weight generators (hypernetworks, weight diffusion, prompt-to-LoRA) exist but are opaque and training-heavy. They do not match wsp's goals.

## 2. RASP (Weiss, Goldberg, Yahav, ICML 2021)

**What it is.** RASP (Restricted Access Sequence Processing) is a programming language intended as a computational model of a transformer *encoder*: attention and feed-forward computation are mapped to simple primitives, and analysing a RASP program gives an upper bound on the layers and heads a transformer needs for the task. The authors also show that a transformer can be trained to mimic a RASP solution. Example programs cover histograms, sorting and Dyck languages. ([arXiv:2106.06981](https://arxiv.org/abs/2106.06981), ICML 2021)

**Primitives.** The base s-ops (sequence operators) are `tokens`, `indices`, `length`. The base functions are `select` (builds a boolean selector matrix from a query s-op, a key s-op and a predicate; this is the attention pattern), `aggregate` (averages values under a selector; this is the attention output) and `selector_width` (counts the selected positions in each row). Elementwise maps stand in for MLPs. ([tech-srl/RASP README](https://github.com/tech-srl/RASP))

**Code.** [tech-srl/RASP](https://github.com/tech-srl/RASP) is an **interpreter and REPL** for its own surface syntax (`.rasp` files), with a library in `RASP_support/rasplib.rasp`. It does **not** produce weights. MIT license, not archived. Last commits are by Gail Weiss on 2024-09-16 ("full pep8 compliance", version 0.1) (checked via GitHub API and `git log`, 2026-11-07).

**Fit for wsp.** RASP is the reference semantics for "transformer programs". Its exact language (from the paper) is not what Tracr compiles. Tracr re-embeds a restricted RASP dialect in Python (see section 3). For wsp, the Python-embedded Tracr dialect is the practical spec language, not the original REPL syntax.

## 3. Tracr (Lindner et al., 2023)

**Paper.** "Tracr: Compiled Transformers as a Laboratory for Interpretability", Lindner, Kramár, Farquhar, Rahtz, McGrath, Mikulik (Google DeepMind). arXiv v1 12 Jan 2023, v5 3 Nov 2023. NeurIPS 2023 spotlight. ([arXiv:2301.05062](https://arxiv.org/abs/2301.05062), [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract.html)). It compiles human-readable RASP programs into standard decoder-only transformers with known structure. The stated purposes are studying superposition and providing ground truth for interpretability methods ([abstract](https://arxiv.org/abs/2301.05062)).

**Pipeline** ([README "How Tracr works"](https://github.com/google-deepmind/tracr); paper section 4):
1. Trace the RASP program into a graph (`rasp_to_graph`).
2. Infer bases: each node gets the set of values it can take. This is a superset found by heuristics, so some values may never occur in practice (paper section 4, step 2).
3. Translate each node into a block of "Craft", an IR that does linear algebra on named subspaces. Elementwise ops become MLPs and select/aggregate pairs become attention heads.
4. Assign blocks to layers **heuristically** (longest-path plus merging). The paper says this is "correct but sometimes suboptimal" (step 4).
5. Build the residual stream as the direct sum of all component spaces, so every s-op gets its own orthogonal subspace for the whole network (step 5).
6. Assemble the weight matrices into a Haiku model. The paper says that porting to another transformer implementation means "reimplementing only step 6" (step 6).
Entry point: `compile_rasp_to_model(program, vocab, max_seq_len, causal=False, compiler_bos, compiler_pad, mlp_exactness=100)` in [`tracr/compiler/compiling.py`](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/compiling.py).

**Encodings** (paper Appendix B; README):
- *Categorical* (default): each value is a one-hot vector in the s-op's subspace.
- *Numerical*: each value is the magnitude of a single dimension. This is needed for mean-style aggregates.
- Every s-op must be annotated as one or the other (`rasp.numerical(...)` / `rasp.categorical(...)`).

**Supported ops** ([`tracr/rasp/rasp.py`](https://github.com/google-deepmind/tracr/blob/main/tracr/rasp/rasp.py)):
- s-ops: `tokens`, `indices`, `Map`, `SequenceMap`, `LinearSequenceMap`, `Full`/`ConstantSOp`, `SelectorWidth`, `Aggregate`.
- selectors: `Select` with a `Comparison` (EQ, LT, LEQ, GT, GEQ, NEQ, TRUE, FALSE).
- `length` is not a primitive. It is built as `SelectorWidth(Select(tokens, tokens, TRUE))` ([README](https://github.com/google-deepmind/tracr)).
- Example library: `make_length`, `make_reverse`, `make_hist`, `make_sort_unique`, `make_sort`, `make_sort_freq`, `make_frac_prevs`, `make_pair_balance`, `make_shuffle_dyck`, `make_count_less_freq`, `make_count`, `make_nary_sequencemap` ([`compiler/lib.py`](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/lib.py)).

**How weights are constructed** (paper Appendix D):
- A categorical→categorical MLP is a lookup table. A categorical `SequenceMap` uses a first layer that maps to an encoding of all input pairs, then a lookup table.
- A numerical MLP is a discretising universal approximator: the first layer buckets the input around the known input values and the second layer maps each bucket to its output. `mlp_exactness` sets the sharpness. The docstring warns that "too large values can cause numerical issues due to large parameter norms. Reasonable values are between 1 and 100" ([`compiling.py`](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/compiling.py)). **These MLPs are approximations, not exact.**
- Attention: W_QK implements the selector plus a rank-one BOS term, scaled by an inverse temperature (100 by default). Softmax then gives about 1/#selected on selected keys. The BOS token is attended to when nothing else is selected, which is why **a BOS token is mandatory** (Appendix B, D.1).
- `SelectorWidth` is an attention layer that attends to BOS and computes 1/(1+w), followed by an MLP that computes w = 1/x − 1 (D.2).
- Causal compilation is available as a flag (D.3).

**Limits.**
- *Restricted RASP.* There are no composite selectors (And/Or/Not of selectors): "Compiling composite Selectors is not supported" ([`expr_to_craft_graph.py`](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/expr_to_craft_graph.py); paper Appendix B, F). Select keys and queries must be categorical. An aggregate's output encoding must match its input encoding. A categorical aggregate must have default None and a numerical one default 0. Non-linear `SequenceMap` is categorical-only and `LinearSequenceMap` is numerical-only ([`validating.py`](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/validating.py)).
- *Silent failures.* A categorical aggregate whose selector width is greater than 1 gives silently wrong outputs. A numerical aggregate supports only 0/1 inputs. These can only be detected at runtime, with `validating.dynamic_validate` ([`validating.py`](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/validating.py)). wsp verification must run this.
- *Size blow-up.* Every variable keeps an orthogonal residual subspace across all layers. Categorical encoding uses one dimension per value, and products of input value sets appear in `SequenceMap` hidden layers. The paper calls the compiled models "sparse and inefficient" (paper sections 5 and A.2). Compressing them is left to a learned projection (below).
- *Expressivity.* Outputs are deterministic sequences, so there are no probabilistic next-token distributions. Attention is binary (paper A.2). The authors state "we will likely never compile fully featured language models in Tracr" (section 7).
- *No layer norm* in compiled models (`layer_norm=False` in [`assemble.py`](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/assemble.py); paper section 4 step 6).
- *No training.* Weights are fully analytic. The only learned part in the paper is compression: a single projection W ∈ R^{D×d} trained with AdamW for 3×10^5 steps while the layer weights stay frozen. It took frac_prevs from D=14 to d=6 "without hurting performance" (paper sections 5.1–5.2, App. E). **The compression code is not released** ([README "Superposition"](https://github.com/google-deepmind/tracr)).

**Repo status.**
- [google-deepmind/tracr](https://github.com/google-deepmind/tracr) is **archived** (read-only). GitHub shows "archived by the owner on Apr 10, 2025" (from search snippet; the API confirms `archived: true`).
- Last commit 2024-02-05 (merges of William-Baker PRs #22/#26). Apache-2.0. About 571 stars and 53 forks (GitHub API, 2026-11-07).
- Dependencies: `jax`, `dm-haiku`, `chex`, `einops`, `networkx`, `numpy` ([`setup.py`](https://github.com/google-deepmind/tracr/blob/main/setup.py)). It is installed from git; there is no PyPI release in setup.py's URL.
- Haiku itself is in **maintenance mode** (bug fixes and JAX compatibility only) and recommends Flax ([dm-haiku README](https://github.com/google-deepmind/dm-haiku)).
- The README calls the Haiku model "documentation-in-code" and expects users to "write or contribute an adapter" to other transformer implementations ([README](https://github.com/google-deepmind/tracr)).

**PyTorch ports.**
- **TransformerLens ships an in-library Tracr converter**: [`transformer_lens/utilities/tracr.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/utilities/tracr.py), with `make_tracr_transformer_bridge_config` and `make_tracr_transformer_bridge_state_dict`, loaded with `TransformerBridge.boot_native(cfg)`.
  - It is duck-typed, so importing TransformerLens does not pull in JAX/Haiku. You still need Tracr+JAX at compile time.
  - Limits stated in the code: categorical output unembed only; `layer_norm=True` raises NotImplementedError.
  - It was added in "Port Tracr demo to native TransformerBridge (#1460)" on 2026-06-30 and fixed in #1525 on 2026-07-21 (GitHub commits API).
  - The demo notebook [`demos/Tracr_to_Transformer_Lens_Demo.ipynb`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb) checks per-layer attention and MLP outputs against Tracr with `np.isclose`.
  - TransformerLens is MIT, actively maintained (pushed 2026-11-06), latest release v4.3.0 on 2026-10-23.
  - This is the most direct off-the-shelf Tracr → PyTorch path.
- The original converter was an informal demo by Neel Nanda, mentioned in a comment on the [Tracr Alignment Forum post](https://alignmentforum.org/posts/2J6fFHQZkWxFcjL6c/tracr-compiled-transformers-as-a-laboratory-for-1) (secondary source, used only for history).
- Forks: 53 in total.
  - The most recently pushed is [recursive-duck/tracr](https://github.com/recursive-duck/tracr) (pushed 2026-11-03). Its commits add binary "bit-encoding" in place of one-hot d_model and a "shrink report" over 160 configs. **Purpose and quality not reviewed.**
  - [FlyingPumba/tracr](https://github.com/FlyingPumba/tracr) (pushed 2024-05-05) is the fork InterpBench depends on (section 6).
  - Others, e.g. [0amp/tracr](https://github.com/0amp/tracr), were not assessed.
  - Source: GitHub forks API, 2026-11-07.

## 4. After Tracr: transformer compilers and program-shaped transformers

### 4.1 RASP-L (Zhou et al., "What Algorithms can Transformers Learn? A Study in Length Generalization")
- **Authors and venue.** Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio, Nakkiran (Apple). arXiv Oct 2023 ([arXiv:2310.16028](https://arxiv.org/abs/2310.16028)). ICLR 2024 ([proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/45ed1a72597594c097152ef9cc187762-Paper-Conference.pdf)).
- **What it is.** RASP-L is a *restricted, causal* RASP for decoder-only models ([paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2310.16028)):
  - Values are int8-bounded.
  - Indices can only be order-compared or moved to their successor or predecessor. Index arithmetic is disallowed, so programs stay uniform across lengths.
  - `aggr` gets min/max aggregations in addition to mean.
  - There is no control flow; programs are straight-line.
  - The core is a `kqv` causal attention primitive.
- **RASP-Generalization Conjecture.** Transformers tend to length-generalize when a *short* RASP-L program solves the task for all lengths ([abstract](https://arxiv.org/abs/2310.16028)).
- **Code.** A numpy implementation of the RASP-L core is printed in the paper (Listing 1, and Appendix Listing 2). The paper says "We plan to release the Python code in these listings on Github" ([ar5iv](https://ar5iv.labs.arxiv.org/html/2310.16028)). **I could not find an official repo** (GitHub search for "rasp-l" turned up nothing relevant).
- **It is not a weight compiler.** It is an interpreter and a learnability theory.
- **wsp fit.** RASP-L's restrictions are a good guide for what an LLM-written spec *should* look like if you later want trained or compressed versions to generalize: causal, no index arithmetic, short programs. The Python listings are small enough to copy as a reference interpreter.

### 4.2 Learning Transformer Programs (Friedman, Wettig, Chen, NeurIPS 2023 oral)
- **Paper.** [arXiv:2306.01128](https://arxiv.org/abs/2306.01128).
- **Direction.** It runs the opposite way from Tracr. The model is *trained* inside a constrained, decompilable subspace and then **decompiled into Python**. Constraints (paper section 3):
  - a *disentangled residual stream*: named discrete variables in orthogonal subspaces, where each module reads variables chosen by one-hot indicator matrices and writes a new variable;
  - categorical attention with a one-to-one predicate matrix, plus numerical attention;
  - MLPs that read ℓ=2 variables and are decompiled into lookup tables by enumeration.
  The discrete weights are learned with Gumbel-softmax relaxation and annealing, and extraction takes the argmax ([PDF](https://arxiv.org/pdf/2306.01128v2)).
- **Tasks.** In-context learning, sort, Dyck, NER, text classification. Results are "on par with standard Transformers of comparable size" ([abstract](https://arxiv.org/abs/2306.01128)).
- **Limits (paper's own Limitations).** It struggles with longer inputs and with finding parsimonious solutions. The authors attribute this mainly to optimization, because hand-written Transformer Programs exist that their method fails to learn. One-to-one predicates make "less than" hard to express.
- **Code.** [princeton-nlp/TransformerPrograms](https://github.com/princeton-nlp/TransformerPrograms). **PyTorch** (README: tested on Python 3.8 and PyTorch 1.13.1). Last commit 2024-05-21 ("Update citation"). Not archived. **No license file** (GitHub API reports no license), so reuse terms are unclear.
- **wsp fit.** This is the relevant prior art for the *verification/round-trip* side: model → program. Its "disentangled residual stream" is essentially the same design as Tracr's residual layout. It is also a PyTorch codebase whose model classes could hold *constructed* (not learned) weights, but that is untested.

### 4.3 ALTA (Shaw et al., TMLR 2025)
- **Paper.** "ALTA: Compiler-Based Analysis of Transformers", Shaw, Cohan, Eisenstein, Lee, Berant, Toutanova (Google DeepMind). [arXiv:2410.18077](https://arxiv.org/abs/2410.18077), v1 Oct 2024, v2 Jun 2025.
- **What it adds over RASP/Tracr.** **Loops**, compilation to **Universal Transformers** (layer-wise weight sharing), and length-invariant constructions for parity and addition. It also solves SCAN without scratchpad ([abstract](https://arxiv.org/abs/2410.18077)). The paper says Tracr "compiles functions over multiple variables ... to dense lookup tables encoded in the MLP parameters", which can explode combinatorially. ALTA instead specifies MLPs as a **sparse set of transition rules**: conjunctive antecedent → set an output variable. With this, SCAN needs fewer than 2,000 hidden dimensions despite more than 10^60 variable combinations ([HTML v2](https://arxiv.org/html/2410.18077v2)).
- **Variables.** Categorical (bounded int), numerical (bucketed, rounded to the nearest bucket in compiled MLPs) and set-valued. Attention is defined by `qkv`, with optional relative positions. A categorical aggregate must select exactly one position.
- **Stated limits.** "limited support for numeric computation", no probabilistic outputs, and compiled models may not resemble learned ones ([HTML v2](https://arxiv.org/html/2410.18077v2)).
- **Trace supervision.** Execution traces from the interpreter are used to train MLPs. This is a finer-grained signal for learnability analysis ([abstract](https://arxiv.org/abs/2410.18077)).
- **Code.** [google-deepmind/alta](https://github.com/google-deepmind/alta) contains the language spec (`framework/`), symbolic interpreter, compiler, a self-contained transformer implementation, and trace tooling. Examples: addition, parity, sort_unique, SUBLEQ, SCAN ([README](https://github.com/google-deepmind/alta)).
  - Dependencies: `absl-py, tensorflow, numpy, jax, apache-beam, scipy` ([setup.py](https://github.com/google-deepmind/alta/blob/main/setup.py)).
  - The compiler emits numpy parameters (`framework/transformer/parameters.py`). The runner uses TensorFlow.
  - Apache-2.0. Last commit 2025-09-22 ("Fix typo"). Not archived. 35 stars (GitHub API, 2026-11-07).
- **wsp fit.** It is the most capable open compiler: loops, weight sharing, and compiled MLPs that are smaller than Tracr's. It has no PyTorch path. Weights are numpy arrays, so a PyTorch loader is a small adapter. **Not verified** in this survey.

### 4.4 Other compiler-like and hand-construction work
- **TracrBench** (Thurnherr & Scheurer, ICML 2024 MI workshop; [arXiv:2409.13714](https://arxiv.org/abs/2409.13714)). 121 RASP programs with compiled transformers, written partly by hand and partly by LLMs and then human-validated. **GPT-4-turbo (20-shot, best-of-5) correctly wrote only 57/101 programs**, and performance degrades quickly with difficulty. The authors also note that Tracr weights differ from trained weights in sparsity and rank (abstract via [mlanthology](https://mlanthology.org/icmlw/2024/thurnherr2024icmlw-tracrbench) and arXiv). **Dataset/repo location not verified.** This is the key data point for wsp's LLM front-end.
- **Neural Decompiling of Tracr Transformers** (Thurnherr & Riesen, ANNPR 2024, LNCS; [arXiv:2410.00061](https://arxiv.org/abs/2410.00061)). A learned decompiler from Tracr weights to RASP: over 30% exact, over 70% functionally equivalent. No code link found.
- **Tracr-Injection** (Vergara-Browne & Soto, ACL Findings 2025; [arXiv:2505.10719](https://arxiv.org/abs/2505.10719)). Distils RASP algorithms into a *pretrained* LM, which yields a residual subspace that decodes to the RASP variables and better OOD behaviour. The paper says code is released. **Repo URL not verified.**
- **PARAT** (Pan et al., "Can Transformers Reason Logically? A Study in SAT Solving"; [arXiv:2410.07432](https://arxiv.org/abs/2410.07432)). A tool for implementing a hand-constructed decoder-only transformer, here a 3-SAT solver with CoT backtracking, as a **PyTorch** model and checking it empirically. The paper says it builds on Tracr/RASP ideas without replacing them ([PDF](https://arxiv.org/pdf/2410.07432)). **No code URL on the arXiv page. Not verified.**
- **Discrete Transformer / "Weights to Code"** (Zhang et al., [arXiv:2601.05770](https://arxiv.org/abs/2601.05770), Jan 2026). Like Transformer Programs, it extracts programs from trained models, using annealed discreteness plus symbolic regression, and also handles continuous intermediates. No code link.

## 5. Hand-constructed / compiled CNN weights

**Headline: I found no general "RASP for CNNs", meaning no compiler from a high-level program to conv weights comparable to Tracr.** An extended web search for compiled or hand-constructed CNN compilers returned only hardware/inference compilers (e.g. [arXiv:1708.00117](https://arxiv.org/pdf/1708.00117), [arXiv:2001.05572](https://arxiv.org/html/2001.05572)). Those compile *already-trained* models to code or hardware, not programs to weights. What exists are scattered analytic constructions:

- **Wavelet scattering networks** (Bruna & Mallat, "Invariant Scattering Convolution Networks", [arXiv:1203.1513](https://arxiv.org/abs/1203.1513)). A cascade of wavelet convolutions, modulus non-linearities and averaging, with filters fixed by wavelet design. Reported state-of-the-art digit and texture classification (with an SVM or PCA classifier on top).
  - **Kymatio** ([arXiv:1812.11214](https://arxiv.org/abs/1812.11214); [kymatio/kymatio](https://github.com/kymatio/kymatio)) implements 1D/2D/3D scattering with NumPy, PyTorch (`kymatio.torch.Scattering2D` is a `torch.nn.Module`), TensorFlow/Keras and JAX frontends ([README](https://github.com/kymatio/kymatio)).
  - BSD-3-Clause. Pushed 2026-06-09 (CI and dependency fixes). Last PyPI release 0.3.0 (2022-09-05) ([PyPI](https://pypi.org/project/kymatio/)).
  - **MPS support not verified.**
  - This is the most mature "CNN with designed, not trained, weights" library. Filters are wavelets, so each weight has a closed-form explanation.
- **Cellular automata as CNNs** (Gilpin, Phys. Rev. E 100, 032402, 2019; [arXiv:1809.02942](https://arxiv.org/abs/1809.02942)). Shows that "any CA may readily be represented" by a network-in-network CNN. The construction is a first conv layer whose receptive field equals the CA neighbourhood, followed by 1×1 layers that implement the rule table. Game of Life is the worked example ([ar5iv](https://ar5iv.labs.arxiv.org/html/1809.02942)). **No code URL on arXiv.** The exact weights are reportedly in the supplement, which I did not verify. This is a template for a "rule table → conv weights" compiler, the CNN analogue of Tracr's categorical MLP lookup tables.
- **"It's Hard for Neural Networks To Learn the Game of Life"** (Springer & Kenyon, [arXiv:2009.01398](https://arxiv.org/abs/2009.01398)). n steps of Life can be implemented in a (2n+1)-layer CNN. Gradient training of near-minimal networks rarely converges, and flipping a single weight's sign can prevent learning. This is direct evidence for wsp's thesis: minimal correct weights exist but are hard to *train* into.
- **Growing Neural Cellular Automata** (Mordvintsev et al., Distill 2020, [distill.pub/2020/growing-ca](https://distill.pub/2020/growing-ca/)). The perception step uses "a 3x3 convolution with a fixed kernel" (Sobel x/y plus identity), and only the update MLP is learned. Code is in [google-research/self-organising-systems](https://github.com/google-research/self-organising-systems) (Apache-2.0, pushed 2026-01-09). This is an example of mixing fixed analytic filters with learned parts.
- **Analytic CNN weights** (Geidarov, "Analytical Calculation of Weights Convolutional Neural Network", arXiv 2505.21557, May 2025; [ResearchGate listing](https://www.researchgate.net/publication/392168725_Analytical_Calculation_of_Weights_Convolutional_Neural_Network)). Computes CNN weights and thresholds directly, without training. **Only seen through a search snippet; contents not verified.**
- **Classic analytic filters** (Sobel, Gabor, DoG/LoG, box/Gaussian blur, morphological ops as min/max-pool). These are textbook constructions and need no new research. wsp can implement them directly as `nn.Conv2d` weights with provenance metadata.

**wsp fit.** For CNNs, wsp would be building the compiler itself. Kymatio covers scattering (fixed wavelet) layers. Gilpin's construction shows how to compile discrete local rules into conv + 1×1 conv. Combining "fixed filter bank" (feature extraction) with "lookup-table 1×1 convs" (Boolean or categorical logic over channels), with Tracr-style one-hot channel encodings, is a natural design. I found no prior art packaging this as a spec language.

## 6. Mechanistic interpretability links

- **Circuits framing.** Tracr's design follows the "mathematical framework" view: the residual stream as shared memory, attention as W_QK/W_OV circuits (Elhage et al. 2021, [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html); cited in the Tracr paper section 2). Compiled models make these objects explicit by construction.
- **Compiled models as ground truth.**
  - Tracr's own stated use cases (paper App. A.1): test cases for probes and attribution, replacing components of trained models with compiled ones, and studying superposition through compression ([arXiv:2301.05062](https://arxiv.org/abs/2301.05062)).
  - **ACDC** (Conmy et al., NeurIPS 2023; [arXiv:2304.14997](https://arxiv.org/abs/2304.14997); [ArthurConmy/Automatic-Circuit-Discovery](https://github.com/ArthurConmy/Automatic-Circuit-Discovery), MIT, pushed 2024-10-01) evaluates circuit discovery on Tracr tasks. The repo contains `acdc/tracr_task/utils.py` and results files `acdc-tracr-reverse-*`, `acdc-tracr-proportion-*` (found by GitHub code search). It also vendors the Tracr→TransformerLens demo notebook.
- **InterpBench** (Gupta, Arcuschin, Kwa, Garriga-Alonso; NeurIPS 2024 Datasets & Benchmarks; [arXiv:2407.14494](https://arxiv.org/abs/2407.14494)).
  - Tracr models are unrealistically sparse. InterpBench trains "semi-synthetic" transformers with **Strict Interchange Intervention Training (SIIT)** so that they implement a known Tracr circuit (or IOI) while being more realistic. Non-circuit nodes are prevented from affecting the output ([abstract](https://arxiv.org/abs/2407.14494)).
  - Code: [FlyingPumba/InterpBench](https://github.com/FlyingPumba/InterpBench), formerly `circuits-benchmark` ([README](https://github.com/FlyingPumba/InterpBench)). MIT. Last push 2024-12-15.
  - PyTorch + TransformerLens 1.19, but still depends on JAX/Haiku via a **Tracr fork** `FlyingPumba/tracr` ([pyproject.toml](https://github.com/FlyingPumba/InterpBench/blob/main/pyproject.toml)).
  - Models are on [Hugging Face cybershiptrooper/InterpBench](https://huggingface.co/cybershiptrooper/InterpBench) (350 files, last modified 2024-10-21, via HF API).
  - Supports ACDC, SP and EAP circuit discovery.
- **TracrBench** (section 4.4): 121 programs as interpretability testbeds. Its authors stress that Tracr weights differ from trained weights in sparsity and rank ([arXiv:2409.13714](https://arxiv.org/abs/2409.13714)).
- **Decompilation and round-trips** (section 4.2, 4.4): Transformer Programs, Neural Decompiling of Tracr, and Discrete Transformer all go from weights back to programs. For wsp, these are possible *verification oracles*: compile → decompile → compare.
- **Compression and superposition**: Tracr section 5 learns W to compress the residual stream and observes superposition. The code is unreleased (section 3).

**wsp fit.** wsp's "every weight is explainable" goal is exactly the property that makes compiled models useful as ground truth. If wsp emits TransformerLens-compatible models with named residual dimensions, it gets the whole MI toolchain (hooks, ACDC, InterpBench evaluation) for free.

## 7. LLM-driven generation of networks and weights

There are two very different families. Only the first one matches wsp's "constructed, explainable" goal.

### 7.1 LLM writes a program, and a compiler produces the weights (symbolic route)
- **TracrBench** ([arXiv:2409.13714](https://arxiv.org/abs/2409.13714)) is the only primary-source measurement I found of LLMs writing compilable RASP. GPT-4-turbo got **57/101** correct with 20-shot prompting and best-of-5 sampling, and accuracy fell with difficulty. The failures were fixed by hand. Implications for wsp:
  - The LLM front-end needs a tight loop: generate → `tracr.compiler.validating` static and dynamic checks → run the RASP evaluator against reference I/O → repair.
  - Few-shot examples should come from `compiler/lib.py` and TracrBench.
  - A newer model is likely better, but **no 2025–2026 re-measurement was found** (extended search, 2026-11-07).
- **Neural surrogate compilation** (Weber, Michel, Renda, Carbin, ICML 2024; [PMLR v235](https://proceedings.mlr.press/v235/weber24b.html); [arXiv:2407.15078](https://arxiv.org/abs/2407.15078)). A **hypernetwork that reads program text (C) and emits neural-surrogate weights**. Reported 1.9–9.5× better data efficiency and 4.3–7.3× fewer training epochs than training surrogates from scratch. These are learned, approximate surrogates, not exact or interpretable constructions. This is the closest published "program text → weights" system outside RASP. **Code not verified.**
- **Inverse direction (weights → program)** with *meta-models*:
  - *Inverse Tracr*, W. Baker, Cambridge MPhil thesis, Sept 2023 ([PDF](https://william-baker.github.io/Inverse_Tracr.pdf)). Code is on OneDrive, not GitHub. The same author's PRs #22 and #26 (computational depth, maintain attributes) were the last commits merged into Tracr ([git log](https://github.com/google-deepmind/tracr/commits/main)).
  - Thurnherr & Riesen 2024 (section 4.4).
  - These show that a learned model can read compiled weights, which is useful as an independent check on wsp output.

### 7.2 Learned weight generators (statistical route)
These generate weights that *work* but are not explainable per weight. They are useful as baselines or for refining constructed weights. They are not a substitute for compilation.
- **HyperNetworks** (Ha, Dai, Le, [arXiv:1609.09106](https://arxiv.org/abs/1609.09106)): one network generates another's weights, trained end-to-end.
- **Weight-space learning / neural functionals.**
  - DWSNets: equivariant layers over MLP weights (Navon et al., ICML 2023, [arXiv:2301.12780](https://arxiv.org/abs/2301.12780)).
  - NFNs: permutation-equivariant neural functionals (Zhou et al., NeurIPS 2023, [arXiv:2302.14040](https://arxiv.org/abs/2302.14040), code [AllanYangZhou/nfn](https://github.com/AllanYangZhou/nfn)). Tasks include predicting generalization, generating sparsity masks, and editing INRs.
  - Survey: Han et al., "A Survey of Weight Space Learning" (Mar 2026, [arXiv:2603.10090](https://arxiv.org/abs/2603.10090)), with a resource list at [Zehong-Wang/Awesome-Weight-Space-Learning](https://github.com/Zehong-Wang/Awesome-Weight-Space-Learning).
- **Diffusion and generative models over weights.**
  - G.pt: a conditional diffusion transformer over checkpoints, conditioned on target loss or return (Peebles et al. 2022, [arXiv:2209.12892](https://arxiv.org/abs/2209.12892), [code](https://www.github.com/wpeebles/G.pt)).
  - p-diff: autoencoder + latent diffusion over parameter subsets (Wang et al. 2024, [arXiv:2402.13144](https://arxiv.org/abs/2402.13144), [code](https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion)).
  - A 2026 position paper argues weight space should be a first-class generative modality ([arXiv:2605.18632](https://arxiv.org/pdf/2605.18632); seen via search only, **not read**).
- **Text/prompt → weights.** Drag-and-Drop LLMs maps task prompts to LoRA weights through a text encoder and a hyper-convolutional decoder. It reports up to 12,000× lower overhead than fine-tuning (Liang et al., Jun 2025, [arXiv:2506.16406](https://arxiv.org/abs/2506.16406), [project](https://jerryliang24.github.io/DnD)). This is "natural language → weights", but it is a trained, opaque generator.

**wsp fit.** All of 7.2 requires large training corpora of trained networks (model zoos). This is the compute wsp wants to avoid, and its output carries no per-weight provenance. wsp should use the 7.1 pattern: **LLM → symbolic spec → deterministic compiler**. 7.2 methods could at most serve as optional comparison points.

## 8. Fit to the wsp pipeline

| Stage | Transformers: existing work | CNNs: existing work | Status |
|---|---|---|---|
| NL intent → spec (LLM) | TracrBench: LLMs write RASP, 57/101 for GPT-4-turbo ([arXiv:2409.13714](https://arxiv.org/abs/2409.13714)) | none found | Partial; needs a repair loop |
| Spec language | RASP ([arXiv:2106.06981](https://arxiv.org/abs/2106.06981)); Tracr's Python RASP dialect ([rasp.py](https://github.com/google-deepmind/tracr/blob/main/tracr/rasp/rasp.py)); RASP-L ([arXiv:2310.16028](https://arxiv.org/abs/2310.16028)); ALTA ([repo](https://github.com/google-deepmind/alta)) | none found (CA rule tables in [Gilpin](https://arxiv.org/abs/1809.02942) come closest) | Transformers: reuse. CNNs: build |
| Spec → weights | Tracr (archived, JAX/Haiku, Apache-2.0); ALTA (TF/JAX/numpy, Apache-2.0) | Kymatio scattering ([repo](https://github.com/kymatio/kymatio)); hand-built filters | Transformers: reuse. CNNs: build |
| → PyTorch model | TransformerLens `utilities/tracr.py` ([file](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/utilities/tracr.py)), categorical outputs only; Transformer Programs model code is PyTorch | Kymatio `kymatio.torch`; plain `nn.Conv2d` | Mostly exists |
| Run on MPS | Not tested by any source | Not tested by any source | **Not verified**; see the float64 note in CLAUDE.md |
| Verification | Tracr `validating.static_validate` / `dynamic_validate`; RASP evaluator as reference; TL demo's per-layer `np.isclose`; decompilers (Transformer Programs, Neural Decompiling) as oracles; InterpBench/ACDC for circuit-level checks | Reference implementation of the filter or CA rule | Pieces exist; no unified harness |

**Pipeline-specific caveats from the sources:**
- Tracr's compiled MLPs approximate the target (`mlp_exactness`), and attention is soft with an inverse temperature of 100 ([compiling.py](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/compiling.py), paper App. D). In float32 on MPS, wsp verification should therefore compare decoded argmax outputs and use tolerances on residuals, with float64 on CPU as the reference.
- Some Tracr miscompilations are silent and only caught by `dynamic_validate` ([validating.py](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/validating.py)). wsp must always run it over the test inputs.
- BOS is mandatory, and `length` costs a SelectorWidth (attention + MLP) ([README](https://github.com/google-deepmind/tracr); paper App. D.2).

## 9. Recommendations: reuse vs build

1. **Transformer spec language: reuse Tracr's Python-embedded RASP. Do not invent a new DSL yet.** It is the de facto standard: TracrBench, InterpBench, ACDC and TransformerLens all consume it. LLMs have seen it, few-shot material exists (`compiler/lib.py`, TracrBench), and it has a reference evaluator. Adopt RASP-L's discipline as an optional lint for programs that should also be learnable or length-general: causal, no index arithmetic, short programs.
2. **Transformer compiler: reuse Tracr as a pinned dependency. Do not fork-and-rewrite at the start.**
   - Path: `compile_rasp_to_model` (JAX/Haiku, CPU) → TransformerLens `make_tracr_transformer_bridge_config` / `make_tracr_transformer_bridge_state_dict` → `TransformerBridge` in PyTorch → `.to("mps")`.
   - This gives hooks and MI tooling for free, plus residual labels (`model.residual_labels`) for the provenance that CLAUDE.md requires.
   - Risks: Tracr is archived and Haiku is in maintenance mode. **Py3.12 + current JAX + Haiku install on M1 is not verified.** Isolate it behind a single wsp module (e.g. `wsp.compilers.tracr_backend`) so it can later be swapped for a native port.
   - If the dependency breaks, porting is cheap. The paper says only step 6 (assembly) is framework-specific, and the Craft IR is numpy-based linear algebra ([README](https://github.com/google-deepmind/tracr)). A wsp-native port of `craft/` + `compiler/` that emits PyTorch directly is a contained project.
3. **Keep ALTA as the second backend** for loops, weight sharing, and large categorical MLPs where Tracr's lookup tables blow up. Its compiled parameters are numpy, so a PyTorch loader is straightforward. Plan time to evaluate it; it has fewer users (35 stars) and a TF dependency.
4. **Do not use Tracr's TL converter for numerical outputs** without extending it (it supports categorical unembed only), and do not use `layer_norm=True` models.
5. **CNNs: build.** There is no compiler to reuse.
   - Reuse Kymatio for wavelet scattering layers.
   - Write analytic filter constructors (Sobel, Gaussian, Gabor, LoG, morphological) as `nn.Conv2d` weights with metadata.
   - Adapt Gilpin's CA→CNN construction for "rule-table → conv + 1×1 conv" logic layers.
   - Borrow Tracr's ideas: named channel subspaces (one-hot categorical channels or scalar numerical channels), a residual/concat layout, and lookup-table MLPs as 1×1 convs.
6. **LLM front-end: build, using TracrBench's lesson.** Treat the LLM as an untrusted program synthesiser inside a loop: generate RASP → static validate → evaluate on reference I/O pairs from a task spec → compile → dynamic validate → run the PyTorch model and compare. Use TracrBench's programs as an eval set (dataset location not yet verified).
7. **Do not adopt learned weight generators** (hypernetworks, p-diff, G.pt, DnD) as the main path. They need model zoos and training and give no per-weight provenance. At most, use them later as baselines.

## 10. Gaps where wsp could contribute

1. **A maintained, PyTorch-native RASP compiler.** Tracr is archived, ALTA is TF/JAX, and TL only converts. Native PyTorch output with MPS-aware numerics (float64 build, float32 run, exactness bounds for `mlp_exactness`/temperature) does not exist.
2. **A CNN compiler / spec language.** There is no "RASP for convolutions" in the literature I found. A typed spec covering filter banks, local rule tables, pooling and channel encodings, compiled to conv weights with provenance, would be new.
3. **Size-efficient compilation without learned compression.** Tracr's orthogonal-subspace layout is "sparse and inefficient", and its compression code is unreleased. ALTA's sparse rules and binary encodings (cf. the recursive-duck fork) point to *analytic* compression that keeps the weights interpretable. Layer allocation is heuristic in Tracr, and the paper itself suggests a constraint solver ([paper A.2](https://arxiv.org/abs/2301.05062)).
4. **An end-to-end NL → RASP → weights benchmark with modern LLMs.** The only published number is GPT-4-turbo in 2024 (TracrBench). Re-measuring with a validate-and-repair loop is a clear, cheap contribution.
5. **Verification harness and provenance format.** A per-weight trace back to a spec element: Tracr has residual labels, but there is no standard export. Equivalence checks across CPU float64 and MPS float32 are also missing. Decompiler round-trips could serve as an independent oracle.
6. **Hybrid constructed + learned models**, e.g. Tracr-Injection-style insertion or InterpBench-style SIIT, as wsp's "optional training" path, kept strictly secondary.

## 11. Unverified items

- RASP-L official code repository. The paper promises one; none was found.
- TracrBench dataset/code location.
- Tracr-Injection, PARAT, Neural Decompiling of Tracr, Weights to Code (2601.05770), and Weber et al. code repositories.
- DWSNets official code URL (not on the arXiv page).
- Whether Tracr installs and runs on Python 3.12 with current JAX/dm-haiku on Apple Silicon. Not tested, because no environment was created for this survey.
- Whether the TransformerLens `TransformerBridge` Tracr models run correctly on MPS.
- Whether Kymatio's torch frontend runs on MPS.
- Gilpin's exact CA→CNN weights (supplementary material not read); no code URL found.
- Geidarov 2025 (arXiv 2505.21557) contents; seen only via a ResearchGate search snippet.
- The 2026 position paper arXiv:2605.18632; seen via search only.
- The Tracr archive date "Apr 10, 2025" comes from a search snippet of the GitHub page. The API confirms only `archived: true`.
- The purpose and quality of the recursive-duck/tracr fork and other Tracr forks.
- Whether any 2025–2026 paper re-measures LLM RASP-writing ability. An extended search found none, but the search was not exhaustive.
