"""Run a harness probe on hand-chosen rows and compare the network's implied
target (up-logit + paddle) to the closed-form reference fold(y + 1.5*vyn*(1-x)/vxn).
This only computes the *task reference*; the network itself runs in the harness."""
import json, subprocess, sys, math

def fold(u):
    m = u % 2.0
    return m if m <= 1 else 2 - m

def ref(x, y, vxn, vyn, p):
    return fold(y + 1.5 * vyn * (1 - x) / vxn)

policy, note, rows = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
out = subprocess.run(["uv", "run", "wsp-pong", "probe", "--agent", "b50k", "--policy", policy,
                      "--obs", json.dumps(rows), "--note", note], capture_output=True, text=True)
if out.returncode:
    print(out.stderr); sys.exit(1)
res = json.loads(out.stdout)
import os, time
os.makedirs("experiments/pong/agents/b50k/probe_results", exist_ok=True)
json.dump({"note": note, "rows": rows, "res": res}, open(f"experiments/pong/agents/b50k/probe_results/{int(time.time())}.json", "w"))
worst = 0
errs = []
for r, lg, a in zip(rows, res["logits"], res["actions"]):
    t_net = lg[2] + r[4]
    t_ref = ref(*r)
    err = t_net - t_ref
    worst = max(worst, abs(err)); errs.append(err)
    print(f"{r} net={t_net:.5f} ref={t_ref:.5f} err={err:+.5f} act={a} stay={lg[1]:.3f}")
print("worst |err|", worst, "mean err", sum(errs)/len(errs), "mean |err|", sum(map(abs, errs))/len(errs), "used", res["used"])
