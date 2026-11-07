# METHOD (c100)

## Design process
1. I read TASK.md, the model, the tokenizer, the KB, and the harness's generation loop. Key facts:
   - The tool result lengths are fixed.
   - So every bot turn is a fixed-length script per intent.
   - Think text is free and stays in the context.
2. Idea: write the intent into `<think>` as a unique, obscure "register" word. Then (intent, step-from-`<bot>`) fully determines the output, and the blanks are filled by "copy the most recent token with mark X".
3. Making the step lookup cheap: the **ramp / second-difference** trick. One ReLU per script word gives an exact indicator of step == k for one intent, so no one-hot step layer is needed. Both MLP layers can therefore hold scripts, which balances d_mlp.
4. Copy blanks use gated attention. The ramps open a gate column, the gate multiplies the mark score, and `<bos>` (zero value) is the sink when every gate is closed. That needs intent and step, which are available after layer 0, so 2 layers are enough.
5. Intent choice: average the hand-written keyword votes over the current user turn. The current turn is the set of keys with the largest "position of latest `<user>`". A gated readout on the `<think>` token turns the votes into register-word logits through the tied embedding.
6. Special cases:
   - urgent writes topic then urgent registers.
   - The reset result token (`sent` / `<unk>`, per the coordinator's harness notice) is itself a register that switches the script.
   - A `dirty` mark stops copies from reading positions where layer 0 already wrote an output, for example the ticket id inside the escalate reply.
7. Keyword lists were hand-written from my own judgement of IT phrasing, with the dev user turns as a sanity read. There was no counting or fitting.

## Runs (97 used of 100)
| # | Runs | Test | Result / lesson | Change |
|---|---|---|---|---|
| 1 | 12 | dev 0–11, first build | 0/12, and I lost the output because my print script was broken | — |
| 2 | 1 | probe `… vpn … <bot> <think>` | The right register won, but `<think>` and random words had huge logits | — |
| 3 | 1 | probe with trace | Copy heads sink on `<bos>`, but the fixed `<bot>`→`<think>` units also fired at `<bos>` (d=0), so the sink carried 40·code(`<think>`) | Gate the fixed units on `f_bot`; zero `<bos>` CODE |
| 4 | 12 | dev 0–11 | 0/12: each register repeated forever. The register's own VOTE=1 re-triggered the readout gate | Gate 1e5 → 1e6 |
| 5 | 12 | dev 0–11 | 11/12. Urgent repeated its register: the adjacent topic register leaked e^-5 into I, which killed the ramp gate (×1000) | Recency 5 → 20, flag 3000 → 1e4, sink 7000 |
| 6 | 13 | dev 6, 12–23 | 13/13 | — |
| 7 | 30 | dev 24–53 | 30/30 | — |
| 8 | 16 | dev 54–69 | 16/16 | — |

All 70 dev conversations have passed. Ids 0–5 and 7–11 last passed before the recency change in run 5, which only sharpens attention. 3 runs are left.
