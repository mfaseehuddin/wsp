# Architecture (arch.py)
Data flow: token ids -> embed table (12-bit code + class flags) -> 6 x [attention heads -> ReLU unit block] on a ~440-channel residual stream whose channels are named slices (weights.py `CH`) -> 12 `out` channels -> x code_table (12 x 4096) -> logits.

Attention is a normal softmax attention with five per-head numbers `hp = (A, shift, R, local, ssum)` (params): score = q.k - A(t-s-shift-dyn)^2 + R s - local*big*|turn_t - turn_s|. A>0 gives exact "t-k tokens back" heads (also a dynamic lag read from channels, used to copy a result into the reply); R gives latest/earliest preference; local restricts to the current turn (turn id = count of `<user>` tokens); ssum=1 turns the head into an exact counter. No branching; only tensor ops.

Why it suits the job: copying, counting and key lookup are attention; the intent and template choice is boolean logic, which ReLU AND units do exactly on 0/1 inputs. ~1.6M params (1.2M zeros in dense blocks), per token ~ 6 layers x (heads 441x16x4 + MLP 441 x <=700). Float32 fine: scores up to ~3e4 (key gain 16000 + recency 12*1023), activations small. Memory 6.5 MB in float32.
Gives up: no learned generalisation; unknown phrasing falls back to the `unknown` reply; whole-context recomputation per token (no cache).
