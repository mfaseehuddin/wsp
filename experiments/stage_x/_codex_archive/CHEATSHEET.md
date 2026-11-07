# Stage X builder API

Read TASK.md. Work only in this directory. You cannot run commands: write files
and ask the coordinator for a public check. It will resume this same session
with feedback. Save a complete candidate before requesting checks.

`weights.py` imports only torch and math; use `torch.float32`. `build()` returns:

```python
cfg = dict(
    d_model=64,
    n_layers=2,
    n_heads=2,
    d_head=8,
    d_mlp=128,
    max_len=16384,
    pos="sinusoidal",
    norm=False,
    bias=True,
    tie_embeddings=False,
)
words = ["hello", "the", "answer"]  # unique exact ASCII words, no whitespace
vocab_size = 272 + len(words)
p = {}
# populate every tensor below; shapes are fixed by cfg and words
groups = [
    dict(
        name="language.token_codes",
        tensor="embed",
        index=None,
        mechanism="...",
        why="...",
    )
]
return cfg, p, groups, words
```

Use basic Python functions/loops/comprehensions, lists, dicts, slices, arithmetic,
range/enumerate/zip/sorted/min/max/sum/len/abs/int/float/str/list/dict/tuple/set,
and explicitly allowed math/torch constructors and tensor methods. No dynamic
attributes, file APIs, random numbers, training, imports beyond torch/math,
classes, while, try, or reflection. All build computations are bounded by a
coordinator timeout. Use local helpers and named constants. No model execution
inside the build. The sanitizer's exact allowlist is in SANITIZER_API.md.

Protocol IDs: PAD=0 BOS=1 USER=2 BOT=3 THINK=4 THINK_END=5 TOOL=6 TOOL_END=7
RESULT=8 RESULT_END=9 END=10 UNK=11 MEMORY=12 MEMORY_END=13 (14/15 reserved).
Byte b is ID 16+b. A word at index j is ID 272+j. Tokenizer scans ASCII
`[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)*`; known exact words become one token,
otherwise their UTF-8 bytes are emitted. Everything else is UTF-8 bytes.
Whitespace is retained as byte tokens. E.g. with words=["hello"], `hello!`
becomes [272, 49]. User text `<tool>` is ordinary bytes/words, never ID 6.

Sequence: BOS, USER, encoded user text, BOT, generated tokens through END;
repeat USER...BOT...END for later turns. Think is optional and unscored. Host
tools accept JSON bytes between TOOL and TOOL_END. JSON may use word tokens
too: decoding concatenates bytes and words without introducing spaces.

Parameter shapes (V=272+len(words), d=d_model, h=n_heads, e=d_head, m=d_mlp):

| Name | Shape |
|---|---|
| embed | V,d |
| pos (only if pos="learned") | max_len,d |
| layers.i.wq / wk / wv | h,d,e |
| layers.i.wo | h,e,d |
| layers.i.bq / bk / bv (bias) | h,e |
| layers.i.bo (bias) | d |
| layers.i.w_in / w_out (m>0) | d,m / m,d |
| layers.i.b_in / b_out (m>0, bias) | m / d |
| layers.i.ln1.g / ln1.b (norm) | d |
| layers.i.ln2.g / ln2.b (norm, m>0) | d |
| lnf.g / lnf.b (norm) | d |
| unembed (unless tied) | d,V |
| b_unembed (bias) | V |

`pos` may be none, learned, sinusoidal. Sinusoidal features in channels 2i,2i+1
are sin(t/10000^(2i/d)), cos(t/10000^(2i/d)); they are added to embed. Learned
positions let you construct your own position codes. No LayerNorm if norm=False.
Per layer: a=LN(x); q=a@wq+bq, k=a@wk+bk, v=a@wv+bv;
x += sum_heads(softmax(causal(q@k.T/sqrt(e)))@v@wo)+bo;
x += relu(LN(x)@w_in+b_in)@w_out+b_out. Finally LNf(x)@U+b_unembed.

Every tensor must be finite float32; max |weight| <=60000; at most 20M scalars.
Full-context float32 KV cache is limited to 512 MiB:
`2 * n_layers * n_heads * d_head * max_len * 4` bytes.
All nonzero elements need provenance. index=None covers a whole tensor;
otherwise index is a list of axes, each an integer, list of integers, or
{"start":0,"stop":12,"step":1}. Groups need real specific explanations,
not simply an arbitrary whole-model label. Every group name appears in docs.

The first candidate may implement a limited subset honestly; extend reusable
mechanisms toward general conversation. A canned-response demonstration is
not a successful general chatbot. There is no hidden pretrained language model.
