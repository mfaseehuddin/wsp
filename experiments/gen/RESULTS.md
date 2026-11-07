# GenGPT stage 0 + explanation audit: results (2026-11-07)

All builders and auditors are Sonnet `wsp-builder` agents. Scores are exact-conversation rates on the hidden test (240 conversations at stage 0, 264 at stage 1). Raw output is in `results_*.json`.

## Builders (stage 0)

| | g10 | g100 | g1000 |
|---|---|---|---|
| Hidden test | 79.2% | **81.3%** | 80.4% |
| Held-out compositions | 81.7% | 81.7% | 76.7% |
| Safety violations | 0 | 0 | 0 |
| Params / nonzero | 1.61M / 113k | 2.00M / 10.0k | **38k / 7.1k** |
| Provenance coverage | 1.0 | 1.0 | 1.0 |
| Dev verified | 6/7 run | 29/29 run | 90/90 |
| Runs / tokens | 9 / 178k | 35 / 214k | 206 / 131k |
| Architecture | custom 6-layer attention, 12-bit id codes | custom 6-layer attention + MLP rules, no dense embed | custom script interpreter: intents → scripts → copy |

## Explanation audits (stage 1: add `wc`, protect `secrets`, new refusal sentence)

Each auditor is compared with its original builder on the same stage-1 hidden set.

| | g10 | g100 | g1000 |
|---|---|---|---|
| Original → audited | 63.3% → 76.5% | 64.0% → **81.1%** | 64.8% → 79.9% |
| wc / protected / unsafe | 12 / 8 / 8 of 12 | 12 / 8 / 12 of 12 | 12 / 8 / 12 of 12 |
| Regressions | 0 | 0 | 0 |
| Nonzero weights changed | +324 | +551 | +46 |
| Auditor tokens / runs | 49k / 27 | 51k / 30 | 38k / 30 |
| Docs score (auditor) | 7/10 | 7/10 | 7/10 |

## Findings

1. **Hand-written lexicons cap generalisation at about 80%, whatever the run budget.** All three fail the same unseen phrasings ("the file with X , what is its K ?", "get rid of F", "take B away from A", "which files do we have ?"). The extra runs bought dev fit (g1000: 90/90) and smaller models, but not a higher hidden score.
2. **Safety mechanisms generalised perfectly.** No model ever had a violation: no escapes, no obeyed injections, no unconfirmed deletes, including unseen paths and unseen injection texts. Structural safety (results can't reach intent or tool routing) is robust.
3. **The mechanisms compose.** Held-out compositions score about the same as the familiar kinds (grep → read → calc in one turn: 12/12, 9/12, 9/12).
4. **The docs explain what each part is well, but not how to add one more of something.** All three auditors independently found the same two gaps: "protect a file" was documented as "treat it as a bad path" (wrong reply), and adding an intent needs exclusions or priorities against the competing intents. Each audit still succeeded with zero regressions, changing 46–551 weights.
5. **Harness issues found:** the vocab included hidden-phrasing words (46 hidden-only words could be spotted; g1000 scanned for them, which wasn't forbidden, and it gave no measurable gain). The fix for stage 2 is to build the vocab from a neutral general word list. Also, path resolution depended on the current directory (fixed).

## GenGPT-opus (built by the coordinator, Opus, 2026-11-07)

This is g1000's audited interpreter plus think-as-working-memory, reference intents, structural intent rules, a "no beats yes" safety tie-break, and follow-ups that use the last number the bot said. It has 54k parameters (7.4k nonzero) and provenance coverage 1.0. Files are in `agents/opus/`.

**Blind fresh test.** 230 conversations at stage 2, with phrasings written by an independent agent that the builder never saw, and no tuning after the first look at per-kind counts:

| | opus | g1000-audit | g100-audit |
|---|---|---|---|
| Whole conversations | **87.0%** | 73.0% | 73.5% |
| Turns | **85.6%** | 67.7% | n/a |
| References (coref) | **6/10** | 0/10 | 0/10 |
| Long sessions / held-out sessions | **4/10 / 3/10** | 1/10 / 1/10 | n/a |
| Safety violations | 0 | 0 | 0 |

Dev at stage 2 is 108/108. The old hidden test scores 100%, but it is contaminated: the builder had seen its failures.
