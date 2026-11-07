# s50: method

## Design reasoning
- The cost is dominated by 470×d_model, so I minimised d. A reply is a fixed sequence, so the next token is a function of only (intent, step n). That removes any need for previous-token or bigram state. Bigram lookups would need step disambiguation anyway, because "." "," "the" repeat inside replies.
- The step comes from 1 position dim plus a head that copies the `<sep>` position: n = t − t_sep. That is exact and linear.
- Intent comes from one head over hand-written keyword lists, with strength tiers and an earliest-wins position penalty. The penalty also suppresses keywords that appear inside replies, so no second layer is needed to separate the query region from the reply region.
- The MLP emits reply_X[n] using intent-gated ReLU ramps, i.e. piecewise-linear interpolation of identity codes. That is 245 neurons, one per reply token.
- Before any pass I checked a few single quantities by hand: attention margins, gate offsets ≥24 when off, and output cosine ≤0.82.
- Keywords were chosen by reading the vocabulary and the dev queries, not from any statistics.

## Forward passes (50/50 used)
1. **eval 14 (ids 0,4,…,52, one per intent).** 14/14 exact on the first build, which confirmed the ramp decoder and sep head for all 14 replies.
2. **eval 14 (13,14,15,25,31,34,35,39,42,46,47,51,53,54).** These are mixed or weak keyword cases: 14/14.
   - *Change:* expanded the keyword lists (vpn: internal/server/sites, slow: boot, mfa: multi, disk: free, monitor: station, OOD: floor, password: log/sign). I also demoted all greeting/thanks words to the weak tier so "hi, my vpn is down" goes to vpn.
3. **probe** "the internet is down" + wifi reply prefix up to "password". Top-1 was "when": the reply-token "password" does not hijack the intent of a weak-only query.
4. **probe** "hello , my vpn is broken". Top-1 was "restart", so the IT intent beats the greeting.
5. **eval 4 (44,45,48,50).** hello/thanks still exact with weak-tier keywords: 4/4.
6. **probe** "thanks , the printer works now". Gives the printer reply ("check"): a strong IT word beats thanks. This is a documented trade-off.
7. **probe** a vpn query with weak "sign" before "vpn", at the final position. Gave `<eos>`, correct.
8. **eval 14 (1,2,3,5,6,7,9,10,11,17,18,19,21,22).** 14/14 on the final model.

Dev verified: 46 distinct queries, 46 exact. 18 of them, plus 4 probes, ran on the final weights. The other 28 ran before the keyword-list edit, which only added keywords and changed tiers.
