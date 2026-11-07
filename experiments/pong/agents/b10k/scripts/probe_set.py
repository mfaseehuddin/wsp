"""Reference crossing heights for hand-picked probe states (task reference, not the network).

crossing = fold(y + 1.5 * b * (1 - x) / a), fold = triangle wave of period 2.
Usage: python probe_set.py            -> prints the JSON obs list
       python probe_set.py OUT.json   -> compares harness probe output with reference
"""
import json, sys

def fold(u):
    m = u % 2.0
    return m if m <= 1.0 else 2.0 - m

# (x, y, a=vx_n, b=vy_n, p)
OBS = [
    [0.0, 0.5, 1.0, 0.0, 0.5],    # flat ball, centred paddle: U=0.5, expect stay
    [0.0, 0.2, 1.0, 0.5, 0.5],    # no bounce: U=0.95
    [0.0, 0.9, 0.5, 1.0, 0.5],    # max slope, extreme U = 3.9
    [0.0, 0.1, 0.5, -1.0, 0.5],   # extreme U = -2.9
    [0.0, 0.0, 0.5, -1.0, 0.5],   # U = -3.0 exactly (range edge)
    [0.0, 1.0, 0.5, 1.0, 0.5],    # U = 4.0 exactly (range edge)
    [0.3, 0.7, 0.6, 0.8, 0.2],    # U = 0.7 + 1.4 = 2.1
    [0.5, 0.4, 0.75, -0.9, 0.8],  # U = 0.4 - 0.9 = -0.5
    [0.9, 0.95, 0.5, 1.0, 0.9],   # late, bounce at top: U = 1.25
    [0.97, 0.05, 0.55, -0.7, 0.1],# very late, small D
    [0.2, 0.33, 0.83, 0.41, 0.66],# generic
    [0.62, 0.71, 0.91, -0.27, 0.12],
    [0.1, 0.5, 0.5, 0.6667, 0.5], # U ~ 2.3
    [0.45, 0.12, 0.67, 0.99, 0.95],
    [0.0, 0.6, 0.7, -0.35, 0.0],  # paddle at bottom wall
    [0.8, 0.5, 1.0, 0.1, 1.0],    # paddle at top wall
]

if len(sys.argv) == 1:
    print(json.dumps(OBS))
else:
    res = json.load(open(sys.argv[1]))
    worst = 0.0
    for o, lg, act in zip(OBS, res["logits"], res["actions"]):
        x, y, a, b, p = o
        U = y + 1.5 * b * (1 - x) / a
        ref = fold(U)
        pred = (lg[2] - lg[0]) / (2 * a) + p  # e = a(fold-p) = (up - down)/2
        err = pred - ref
        worst = max(worst, abs(err))
        want = 2 if ref - p > 0.015 else (0 if ref - p < -0.015 else 1)
        print(f"U={U:+.4f} ref={ref:.5f} pred={pred:.5f} err={err:+.2e} act={act} want={want}")
    print("max |err| =", worst)
