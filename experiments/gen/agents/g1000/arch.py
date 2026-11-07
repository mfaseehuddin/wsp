import torch
import torch.nn.functional as F

ZERO = 0
ONE = 1
NEG = -1


def forward(params, tokens):
    p = params
    B, T = tokens.shape
    pos = p['k.pos'][:T].unsqueeze(ZERO).expand(B, T)
    tid = tokens.to(pos.dtype)
    zero = p['k.zero']
    half = p['k.half']
    one = p['k.onef']
    lex = p['lex.feat'][p['lex.id'][tokens].long()]

    def ft(name):
        return (lex * p['col.' + name]).sum(NEG)

    def last(flag):
        return torch.cummax(torch.where(flag > half, pos, p['k.neg1']), dim=ONE).values

    def shift(x):
        return torch.cat([zero.expand(B, ONE), x[:, :NEG]], dim=ONE)

    def at(x, idx):
        i = torch.minimum(torch.maximum(idx, zero), pos[:, NEG:]).round().long()
        return torch.gather(x, ONE, i)

    def eq(a, b):
        return ((a - b).abs() < half).to(pos.dtype)

    nexttid = torch.cat([tid[:, ONE:], zero.expand(B, ONE)], dim=ONE)
    user, bot, eres, res = ft('user'), ft('bot'), ft('eres'), ft('res')
    pu, pb, pdec = last(user), last(bot), last(bot + eres)
    inuser = (pu > pb).to(pos.dtype)
    inres = (last(res) > last(eres)).to(pos.dtype)
    pq, pk = pos.unsqueeze(2), pos.unsqueeze(ONE)
    # MU: query t sees the tokens of the CURRENT user turn only.  MT: the tokens of the current bot turn.
    MU = ((pk > pu.unsqueeze(2)) & (pk <= pq)).to(pos.dtype) * inuser.unsqueeze(ONE)
    MT = ((pk > pb.unsqueeze(2)) & (pk <= pq)).to(pos.dtype)

    # ---- user bag -> intent
    bag = torch.bmm(MU, lex @ p['bag.sel'])
    qpos = last(ft('q') * (ONE - inuser) * (ONE - inres))
    pend = ((qpos + p['k.two']) == pu).to(pos.dtype) * (qpos > half).to(pos.dtype)
    z = torch.cat([bag, pend.unsqueeze(NEG)], dim=NEG) @ p['ind.sel'] - p['ind.thr']
    x = F.relu(z) - F.relu(z - one)
    sc = F.relu(x @ p['flow.and'] + p['flow.bias']) * p['flow.prio'] + p['flow.base']
    fl = torch.softmax(sc * p['k.fgain'], dim=NEG)

    # ---- user slots (n-th flagged token of current user turn)
    cols = []
    for i in range(p['slot.n'].shape[ZERO]):
        flag = (lex * p['slot.flag'][i]).sum(NEG)
        mk = MU * flag.unsqueeze(ONE)
        rank = torch.cumsum(mk, dim=NEG)
        tot = mk.sum(NEG, keepdim=True)
        want = torch.where(p['slot.n'][i] > half, p['slot.n'][i], tot)
        sel = mk * eq(rank, want)
        val = tid * p['slot.val'][i][ZERO] + ft('opid') * p['slot.val'][i][ONE] + pos * p['slot.val'][i][2]
        cols.append((sel * val.unsqueeze(ONE)).sum(NEG))
    SL = torch.stack(cols, dim=NEG)

    # ---- turn-relative result machinery
    rr = torch.cumsum(MT * res.unsqueeze(ONE), dim=NEG)
    c = (MT * eres.unsqueeze(ONE)).sum(NEG)

    def res_tok(n):
        sel = MT * res.unsqueeze(ONE) * eq(rr, n.unsqueeze(2))
        return (sel * nexttid.unsqueeze(ONE)).sum(NEG), (sel * (pos + one).unsqueeze(ONE)).sum(NEG)

    lstart = (shift(res + ft('semi')) > half).to(pos.dtype)

    def look(K, r):
        sel = MT * (inres * lstart).unsqueeze(ONE) * eq(rr, r.unsqueeze(2)) * eq(tid.unsqueeze(ONE), K.unsqueeze(2))
        return (sel * nexttid.unsqueeze(ONE)).sum(NEG), sel.sum(NEG)

    sv = SL @ p['sv.fromslot']
    ansflag = last(ft('is') * shift(ft('answer')))
    ANS = at(nexttid, ansflag) * (ansflag > half).to(pos.dtype)
    HIT = res_tok(torch.ones_like(c))[ZERO]
    PREVF = at(tid, qpos - one) * (qpos > half).to(pos.dtype)
    RA = (fl @ p['flow.ra']).round()
    RB = (fl @ p['flow.rb']).round()
    K1 = (SL * p['sl.k1']).sum(NEG)
    K2 = (SL * p['sl.k2']).sum(NEG)
    V1, found1 = look(K1, RA)
    V2, found2 = look(K2, RB)
    RES0, rpos = res_tok(c)
    sv = sv + ANS.unsqueeze(NEG) * p['sv.ans'] + HIT.unsqueeze(NEG) * p['sv.hit'] + PREVF.unsqueeze(NEG) * p['sv.prevf']
    sv = sv + V1.unsqueeze(NEG) * p['sv.v1'] + V2.unsqueeze(NEG) * p['sv.v2'] + RES0.unsqueeze(NEG) * p['sv.res0'] + p['k.tok2'] * p['sv.two']
    # operands of calc: pick two, swap by text order / 'subtract B from A'
    x1 = torch.einsum('btf,fs,bts->bt', fl, p['opd.a'], sv)
    x2 = torch.einsum('btf,fs,bts->bt', fl, p['opd.b'], sv)
    pk1 = (SL * p['sl.pk']).sum(NEG)
    pn1 = (SL * p['sl.pn']).sum(NEG)
    litfirst = (pn1 < pk1).to(pos.dtype)
    rev = bag @ p['bag.rev']
    rev = rev - F.relu(rev - one)
    swp = (fl @ p['opd.swap']) * (rev + litfirst - p['k.two'] * rev * litfirst)
    sv = sv + (x1 + swp * (x2 - x1)).unsqueeze(NEG) * p['sv.arga'] + (x2 + swp * (x1 - x2)).unsqueeze(NEG) * p['sv.argb']

    # ---- status of the latest result
    eqs = (RES0.unsqueeze(NEG) == p['st.ids']).to(pos.dtype)
    hasres = (c > half).to(pos.dtype)
    bad = eqs.sum(NEG)
    nokey = (ONE - torch.minimum(found1, one)) * (ONE - bad) * hasres
    st = torch.cat([((ONE - bad) * (ONE - nokey)).unsqueeze(NEG), eqs, (nokey).unsqueeze(NEG)], dim=NEG)

    # ---- script: call phase and reply phase share k = tokens since the decision point
    k = pos - pdec
    cidx = c.round().unsqueeze(NEG) == p['k.cr']
    jh = (k.round().unsqueeze(NEG) == p['k.jr']).to(pos.dtype)
    coh = cidx.to(pos.dtype)
    tl = torch.einsum('btf,btc,fcj->btj', fl, coh, p['call.lit'])
    ts = torch.einsum('btf,btc,fcjs->btjs', fl, coh, p['call.slot'])
    has = (tl[..., ZERO] > half).to(pos.dtype) * (ONE - bad)
    call_t = (tl * jh).sum(NEG) + (torch.einsum('btjs,btj->bts', ts, jh) * sv).sum(NEG)

    rt = torch.einsum('btf,btu,fur->btr', fl, st, p['rt.map'])
    kh = (k.round().unsqueeze(NEG) == p['k.kr']).to(pos.dtype)
    tmpl_t = torch.einsum('btr,rk,btk->bt', rt, p['tmpl.lit'], kh)
    tmpl_t = tmpl_t + (torch.einsum('btr,rks,btk->bts', rt, p['tmpl.slot'], kh) * sv).sum(NEG)
    k0 = rt @ p['tmpl.k0']
    cm = F.relu(k - k0 + one) - F.relu(k - k0)
    src = rpos + (k - k0)
    ctid = at(tid, src)
    cend = at(eres, src)
    copy_t = ctid + cend * (p['k.period'] - ctid)
    reply_t = (one - cm) * tmpl_t + cm * copy_t
    target = has * call_t + (one - has) * reply_t
    endf = ((ft('dot') + ft('q')) > half).to(pos.dtype)
    target = endf * p['k.end'] + (one - endf) * target
    d = p['k.vid'].view(ONE, ONE, NEG) - target.unsqueeze(NEG)
    return -p['k.gain'] * torch.minimum(d.abs(), p['k.cap'])
