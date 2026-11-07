# c1 method

Budget: 1 run and 250k tokens. With a single run, the design had to be correct by construction and checkable without running the model.

## Design process
1. I read the task, the model, the tokenizer, the knowledge base, the harness generation loop and the 70 dev user turns. I read the dev phrasings only to see what they look like. I wrote the keyword lists myself and did not derive them by counting.
2. Key choices:
   - Represent every word the bot needs as a one-hot dimension, and use tied embeddings so that the same dimension serves as the output.
   - Choose intent by having attention pick the strongest keyword in the current user turn.
   - Use head 1 of layer 0 to find where the turn starts. Making the earlier turns and the bot's own words invisible to the keyword heads relies on this.
   - Let an MLP play the replies as n-gram scripts. The key is (intent, current word) and adds the previous word only where that is ambiguous.
   - Fill slots with "latest X" copy heads.
   - Write the FAQ topic inside `<think>` so escalation can find it later.
3. Everything lands in 2 layers. Layer 0 handles position, segment and copy sources. Layer 1 handles intent, think-topic, state and the scripts. All attention scores have margins of at least 12 nats, and most are over 60. There is no LayerNorm, so every quantity is linear and can be checked by hand.
4. `check.py` is a symbolic check of the script table, not a forward pass. It expands every script with every slot value and checks that exactly the right unit fires at every position the model predicts. The result is 0 conflicts across 597 script units.
5. A static review found a phrasing risk: "suspicious email" would go to email. I added a strong-synonym tier (suspicious, weird, fake, deleted, recover, hire, starter, onboarding) that beats topic names.
6. I noticed that `unknown` (a reset_password result) is not in the vocabulary, so the unknown-user branch is keyed on `<unk>`.

## Runs
| # | What | Result | Learned / changed |
|---|---|---|---|
| 1 | `eval --ids 3` (escalate-recall: FAQ with think, escalate with think-topic head, id copy and team, then recall) | 1/1 passed, 3/3 turns | Confirmed end to end that the previous-token, segment, keyword, think-topic and id-copy heads and the script MLP work as computed. No changes. |
