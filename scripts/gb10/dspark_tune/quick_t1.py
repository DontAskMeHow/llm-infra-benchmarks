#!/usr/bin/env python3
"""Quick T1 analysis: per-N summary from timeline + metrics."""
import json, sys, os, statistics

root = sys.argv[1]
for d in sorted(os.listdir(root)):
    tl_path = os.path.join(root, d, "timeline.jsonl")
    if not os.path.isfile(tl_path): continue
    rows = [json.loads(l) for l in open(tl_path)]
    warm = [r for r in rows if r.get("kind")=="warm" and r.get("status")=="ok" and r.get("ts_first")]
    if not warm: print(d, "EMPTY"); continue
    n = len(warm)
    u = [r.get("usage", {}) for r in warm]
    ct = sum(x.get("completion_tokens",0) for x in u)
    tpots = [(r["ts_last"]-r["ts_first"])/max(1,r.get("usage",{}).get("completion_tokens",1)-1) for r in warm]
    ttfts = sorted([r["ts_first"]-r["ts_sent"] for r in warm])
    span = max(r["ts_last"] for r in warm) - min(r["ts_sent"] for r in warm)
    # per-pos acceptance from metrics.raw
    mpath = os.path.join(root, d, "metrics.raw")
    pos_acc = {}
    if os.path.isfile(mpath):
        for line in open(mpath, errors="replace"):
            if "spec_decode_num_accepted_tokens_per_pos" in line:
                try:
                    ts_str, rest = line[1:].rsplit(" ", 1)
                    val = float(rest)
                    pos_str = line.split('position="')[1].split('"')[0]
                    pos_acc.setdefault(pos_str, []).append(val)
                except: pass
    acc_str = ""
    for p in range(7):
        pts = pos_acc.get(str(p), [])
        if len(pts) >= 2:
            delta = pts[-1] - pts[0]
            acc_str += " p%d=%d" % (p, delta)
    print("%-5s reqs=%2d agg_tps=%6.1f tpot=%5.1fms ttft_p95=%5.3fs%s" % (
        d.replace("A-","N"), n, ct/span if span else 0,
        statistics.mean(tpots)*1000, ttfts[min(n-1, int(n*0.95))], acc_str))
