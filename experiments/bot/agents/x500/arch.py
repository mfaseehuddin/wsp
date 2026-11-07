"""x500 architecture: a "rule-memory machine" built only from tensor ops.

Four stages, all reading named tables in `params`:
  1. LEXICON   token id -> feature vector (soft range-match against rows of a table).
  2. COUNTERS  where is the current bot turn, how far into the reply are we (cummax/cumsum).
  3. HEADS     five hard-attention "copy" heads (latest topic / ticket id / service / result word / user name).
  4. PROGRAM   a memory of (script, step) rows; the matching row writes a vote for the next token.
"""

import torch
import torch.nn.functional as F


def bump(a):
    # triangular kernel: 1.0 exactly at 0, 0.0 at distance >= 1
    return F.relu(inv(a.abs()))


def col(a):
    return a.unsqueeze(-1)


def inv(a):
    return 1 - a


def sat(a):
    return a.clamp(0, 1)


def core(P, tokens):
    dt = P["plan_bias"].dtype
    B, T = tokens.shape
    V = 2000
    x = tokens.to(dt)
    pos = torch.arange(T, device=tokens.device).to(dt)
    vax = torch.arange(V, device=tokens.device).to(dt)

    # ---- 1. lexicon: every token id -> 50 features (32 evidence dims + 18 flags)
    xe = x.unsqueeze(-1)
    box = F.relu(inv(F.relu(P["lex_lo"] - xe) + F.relu(xe - P["lex_hi"])))
    onehot = torch.eq(col(P["lex_dim"]), P["dim_grid"]).to(dt)
    feat = box @ (col(P["lex_w"]) * onehot)
    ev, ex = feat.split(32, -1)
    g1, g2, g3 = ex.split(6, -1)
    u, bt, th, te, _a, _b = g1.unbind(-1)
    _c, rse, _d, PI, TM, ftop = g2.unbind(-1)
    fid, fsvc, fres, fname, emit, vt2 = g3.unbind(-1)

    # ---- 2. counters
    bpos = torch.cummax(pos * bt, -1).values  # last <bot>
    upos = torch.cummax(pos * u, -1).values  # last <user>
    epos = torch.cummax(pos * te, -1).values  # last </think>
    rpos = torch.cummax(pos * (te + rse), -1).values  # last </think> or </result>
    apos = torch.cummax(pos * rse, -1).values  # last </result>
    post = sat(epos - bpos)  # 1 once this turn's think is closed
    after = sat(apos - bpos)  # 1 once a tool result arrived this turn
    pre = inv(post)
    k = pos - rpos  # steps since the last reset marker
    ithink = torch.cumsum(th, -1) - torch.cumsum(te, -1)  # 1 while inside <think>

    # ---- bag of keyword evidence over the current user message
    M = ((pos > col(upos)) & (pos < col(bpos))).to(dt)
    m = M @ ev + P["plan_bias"]

    # ---- think phase: pick the plan token (and the topic after "urgent")
    bumpP = bump(vax - col(P["plan_id"]))
    beta = P["beta"]
    vote1 = torch.softmax(beta * m, -1) @ bumpP
    vote2 = torch.softmax(beta * m + P["topic_mask"], -1) @ bumpP
    lg = (pre * th).unsqueeze(-1) * vote1 + (pre * vt2).unsqueeze(-1) * vote2
    lg = lg + (pre * sat(emit)).unsqueeze(-1) * bump(vax - col(emit))

    # plan identity token = 2 tokens after <bot>
    idx = torch.minimum(bpos + 2, pos).long()
    PI1 = PI.gather(1, idx)

    # ---- 3. copy heads
    fl = torch.stack((ftop, fid, fsvc, fres, fname), -1)
    ok = fl * (1 - (col(ithink) - P["head_req"]).abs())
    att = P["att"]
    fut = (pos.view(1, T) > pos.view(T, 1)).to(dt)
    score = att[0] * ok.unsqueeze(1) + att[1] * pos.view(1, 1, T, 1) - att[2] * fut.unsqueeze(-1)
    w = torch.softmax(score, 2)
    vid = torch.einsum("btsh,bs->bth", w, x)
    wt = w.unbind(-1)
    vteam = (wt[0] * TM.unsqueeze(1)).sum(-1)
    vtop, vidn, vsvc, vres, vname = vid.unbind(-1)
    sv = torch.stack((vtop, vteam, vidn, vsvc, vres, vname), -1)

    # ---- 4. program memory
    pm = bump(col(PI1) - P["rom_plan"])
    am = bump(col(after) - P["rom_after"])
    km = bump(col(k) - P["rom_k"])
    need = sat(P["rom_res"])
    cond = inv(need * inv(bump(col(vres) - P["rom_res"])))
    act = col(post) * pm * am * km * cond
    litmask = inv(sat(P["rom_slot"]))
    lg = lg + (act * litmask) @ bump(vax - col(P["rom_tok"]))
    OHs = torch.eq(col(P["rom_slot"]), P["slot_grid"]).to(dt)
    g = act @ OHs
    lg = lg + (col(g) * bump(vax - col(sv))).sum(2)
    return {"logits": lg, "m": m, "bpos": bpos, "upos": upos, "post": post, "k": k, "PI1": PI1, "vres": vres}


def forward(P, tokens):
    return core(P, tokens)["logits"]


def trace(P, tokens):
    return core(P, tokens)
