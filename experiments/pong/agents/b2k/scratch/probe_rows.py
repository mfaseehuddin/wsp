"""Make probe rows and the reference crossing height (task formula, not the network)."""
import json, sys

def fold(u):
    m = u % 2.0
    return m if m <= 1.0 else 2.0 - m

def ref_t(x, y, v, c):
    return fold(y + 1.5 * c * (1 - x) / v)

def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b; r += f * (i % b); i //= b
    return r

rows = [
    [0.0, 0.5, 0.5, 1.0, 0.5], [0.0, 0.5, 0.5, -1.0, 0.5],
    [0.0, 0.0, 0.5, 1.0, 0.2], [0.0, 1.0, 0.5, -1.0, 0.8],
    [0.0, 0.3, 1.0, 1.0, 0.5], [0.0, 0.7, 0.7071, -0.6, 0.5],
    [0.96, 0.4, 0.5, 1.0, 0.45], [0.99, 0.02, 0.5, -1.0, 0.1],
]
for i in range(1, 41):
    x = 0.98 * halton(i, 2); y = halton(i, 3); v = 0.5 + 0.5 * halton(i, 5)
    c = 2 * halton(i, 7) - 1; p = halton(i, 11)
    rows.append([round(x, 4), round(y, 4), round(v, 4), round(c, 4), round(p, 4)])
if sys.argv[1:] == ["rows"]:
    print(json.dumps(rows))
else:
    print(json.dumps([ref_t(r[0], r[1], r[2], r[3]) for r in rows]))
