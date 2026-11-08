# Example models (read-only references)

Hand-written models from earlier wsp experiments. Every weight in them was written by an AI agent, with no training.
Study them for **techniques**: they are not solutions to copy. Each folder has the model code and the agent's
`EXPLANATION.md`. Read the explanation first; it is shorter. They target other harnesses, so they won't run in
`wsp-pure` as they are.

| folder | what it is | result | techniques worth studying |
|---|---|---|---|
| `chat_ncm_nco60/` | Open-ended chat, Next-Concept Model, 2 levels, 6 layers. **This same task** in an earlier round, where the agent could test locally and used Qwen embedding geometry. | 7/33 dev checks | RoPE offset kernels (previous-token heads); attention counters with a sink (turn and sentence counting); concept directions; copying a span by attention hops; template pieces + copied span |
| `deep_ncm_ncd60/` | Open-ended chat, 4-level NCM (act → content word → phrase skeleton → tokens) | 4/33 dev checks | stating and probing what each level means; sentence-id codes; picking a content word; a no-match sink for "I don't know" |
| `helpdesk_ncm_nco/` | Help-desk bot, 2-level NCM: a plan code, then all tokens in parallel | **97.3%** on 448 unseen messages | keyword-voting intent head; attention-to-sink presence detectors; an exact (plan, position) → token layout bank; copy heads for a name and digits |
| `adder_circuit_o60/` | An exact 12-digit adder written into Qwen2.5-0.5B as 10 inserted layers | **100%** on 960 hidden problems | constant-axis trick to make RMSNorm linear; fixed-offset attention via RoPE; exact carry look-ahead in MLPs; writing a large vector to force an output token |

Notes:
- `chat_ncm_nco60` and `deep_ncm_ncd60` used Qwen2.5-0.5B's embedding or layers. **This round allows no pretrained
  weights**: everything you write comes from your own reasoning.
- In `wsp-pure` the NCM family is the same model as `src/wsp/ncmchat/model.py`. `build()` returns
  `(family, cfg, params, groups)` instead of `(cfg, params, groups)`.
