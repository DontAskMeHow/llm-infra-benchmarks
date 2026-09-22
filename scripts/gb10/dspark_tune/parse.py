#!/usr/bin/env python3
"""Разбор результатов dspark-tune: timeline + events + metrics.raw -> CSV/таблица.

Использование:
  python3 parse.py /tmp/dspark-tune/20260912-A   # все run-каталоги внутри
Пишет summary.csv в корень каталога и печатает таблицу.
"""
import csv
import glob
import json
import os
import statistics
import sys


def load_events(path):
    evs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                evs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return evs


def load_metrics(path):
    """[@ts name{labels} val] -> {key: [(ts,val)]}; ключ = name[:pos] — по позиции
    отдельные серии для per-pos метрик, иначе одно суммируемое имя."""
    series = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("@"):
            continue
        try:
            head, vstr = line[1:].rsplit(" ", 1)
            ts_str, rest = head.split(" ", 1)
            ts, val = float(ts_str), float(vstr)
            name, _, lbls = rest.partition("{")
        except (ValueError, IndexError):
            continue
        pos = ""
        if 'position="' in lbls:
            pos = lbls.split('position="')[1].split('"')[0]
        key = name + (":" + pos if pos else "")
        series.setdefault(key, []).append((ts, val))
    return series


def _share_rows(tl):
    f = open(tl, encoding="utf-8")
    return [json.loads(l) for l in f if l.strip().startswith("{")]


def _delta(series, t0, t1):
    pts = [p for p in series if t0 - 1 <= p[0] <= t1]
    if len(pts) < 2:
        return None, None
    return pts[0][1], pts[-1][1]


def parse_run(rundir, out, skip_start=5.0):
    tl = _share_rows(os.path.join(rundir, "timeline.jsonl"))
    evs = load_events(os.path.join(rundir, "events.jsonl"))
    starts = [e["ts"] for e in evs if e["event"] == "phase_start"]
    stops = [e["ts"] for e in evs if e["event"] == "phase_stop"]
    aborted = any(e.get("aborted") for e in evs if e["event"] == "phase_stop")
    t0 = min(starts) if starts else min((r["ts_sent"] for r in tl), default=0)
    t1 = max(stops) if stops else max((r.get("ts_last") or r["ts_sent"] for r in tl), default=t0)
    w0 = t0 + skip_start
    if w0 >= t1:
        w0 = t0

    warm = [r for r in tl if r.get("kind") == "warm" and r.get("status") == "ok"
            and r.get("ts_first") and w0 <= r["ts_sent"] <= t1]
    cold = [r for r in tl if r.get("kind") == "cold" and r.get("status") == "ok"
            and r.get("ts_first") and w0 <= r["ts_sent"] <= t1]

    def stats(rows):
        if not rows:
            return dict(n=0, ttft_mean=0, ttft_p50=0, ttft_p95=0,
                        tpot_mean=0, tpot_p95=0, first_last_span=0,
                        cache_ratio=0, pt=0, ct=0)
        ttfts = sorted(r["ts_first"] - r["ts_sent"] for r in rows)
        usages = [r.get("usage", {}) or {} for r in rows]
        ct = sum(u.get("completion_tokens", 0) for u in usages)
        td = max(r["ts_last"] - r["ts_first"] for r in rows) or 1e-6
        tpots = [(r["ts_last"] - r["ts_first"]) /
                 max(1, (r.get("usage", {}).get("completion_tokens", 1) - 1))
                 for r in sorted(rows, key=lambda x: x["ts_sent"])]
        cached = sum((u.get("prompt_tokens_details", {}) or {}).get("cached_tokens", 0)
                     for u in usages)
        pt_total = sum(u.get("prompt_tokens", 0) for u in usages)
        return dict(n=len(rows),
                    ttft_mean=statistics.mean(ttfts),
                    ttft_p50=statistics.median(ttfts),
                    ttft_p95=ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.95))],
                    tpot_mean=statistics.mean(tpots),
                    tpot_p95=statistics.median(sorted(tpots, reverse=True)[:max(1, len(tpots) // 20)]),
                    first_last_span=(max(r["ts_last"] for r in rows) -
                                     min(r["ts_sent"] for r in rows)),
                    cache_ratio=(cached / pt_total if pt_total else 0),
                    pt=pt_total, ct=ct)

    ws, cs = stats(warm), stats(cold)
    wspan = ws["first_last_span"] or (t1 - w0)
    cspan = cs["first_last_span"]

    m = load_metrics(os.path.join(rundir, "metrics.raw"))
    row = dict(run=os.path.basename(rundir), aborted=aborted, dur=t1 - t0)
    names = ["iteration_tokens_total_count", "iteration_tokens_total_sum",
             "spec_decode_num_draft_tokens_total", "spec_decode_num_drafts_total",
             "spec_decode_num_accepted_tokens_total",
             "num_preemptions_total", "prefix_cache_queries_total",
             "prefix_cache_hits_total", "prompt_tokens_total",
             "generation_tokens_total", "num_requests_waiting",
             "num_requests_running", "kv_cache_usage_perc"]
    for name in names:
        a, b = _delta(m.get("vllm:" + name, []), w0, t1)
        row["d_" + name] = (b - a) if (a is not None and a <= b) else None
    for pos in range(7):
        a, b = _delta(m.get(
            "vllm:spec_decode_num_accepted_tokens_per_pos_total:" + str(pos), []),
            w0, t1)
        row["acc_pos%d" % pos] = (b - a) if (a is not None and a <= b) else None
    row["acc_pos1_6"] = sum(v for v in [row.get("acc_pos%d" % p) for p in range(1, 7)]
                            if v)
    iters = row.get("d_iteration_tokens_total_count") or 0
    if iters and (t1 - w0) > 0:
        row["ms_per_iter"] = 1000.0 * (t1 - w0) / iters
        row["tokens_per_iter"] = (row.get("d_iteration_tokens_total_sum") or 0) / iters
    else:
        row["ms_per_iter"] = row["tokens_per_iter"] = None
    dra = row.get("d_spec_decode_num_drafts_total") or 0
    drt = row.get("d_spec_decode_num_draft_tokens_total") or 0
    row["drafts_per_decode"] = drt / dra if dra else None  # K-check: ~7
    row["accept_total"] = row.get("d_spec_decode_num_accepted_tokens_total")
    pq = row.get("d_prefix_cache_queries_total") or 0
    ph = row.get("d_prefix_cache_hits_total") or 0
    row["prefix_hit%"] = 100.0 * ph / pq if pq else None
    row.update(n_warm=ws["n"], ttft_mean=ws["ttft_mean"], ttft_p50=ws["ttft_p50"],
               ttft_p95=ws["ttft_p95"], tpot_mean=ws["tpot_mean"],
               warm_agg_tps=(ws["ct"] / wspan if wspan else 0),
               warm_cache_ratio=ws["cache_ratio"], warm_pt=ws["pt"],
               warm_ct=ws["ct"],
               n_cold=cs["n"], cold_ttft_p95=cs["ttft_p95"],
               cold_drain_tps=(cs["pt"] / cspan if cspan else 0),
               cold_pt=cs["pt"])
    out.append(row)
    return row


def main():
    root = sys.argv[1]
    skip = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    rows = []
    for rundir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isfile(os.path.join(rundir, "timeline.jsonl")):
            continue
        try:
            rows.append(parse_run(rundir, rows, skip))
        except Exception as e:
            print("ERUN %s: %s" % (rundir, e), file=sys.stderr)
    if not rows:
        print("no runs in", root)
        return
    keys = [k for k in rows[0].keys() if not k.startswith("first_last")] + []
    with open(os.path.join(root, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    hdr = ("run     Nwarm  ms/iter  tok/iter  drafts/decode  acc0    acc1-6  "
           "warmTTFTp95  tpot   warmtps  cache%  coldN  coldTTFTp95  abd")
    print(hdr)
    for r in rows:
        n = r["run"].split("-")[-1] if "-" in r["run"] else r["run"]
        print("%-7s %4d  %7.2f  %8.1f  %13s  %6s  %6s  %9.3f  %6.3f  %7.1f  %5.1f  %5d  %10.3f  %s"
              % (n, r["n_warm"],
                 r["ms_per_iter"] or 0, r["tokens_per_iter"] or 0,
                 ("%.2f" % r["drafts_per_decode"]) if r["drafts_per_decode"] else "-",
                 ("%d" % r["acc_pos0"]) if r["acc_pos0"] is not None else "-",
                 ("%d" % r["acc_pos1_6"]) if r["acc_pos1_6"] is not None else "-",
                 r["ttft_p95"], r["tpot_mean"], r["warm_agg_tps"],
                 100.0 * r["warm_cache_ratio"], r["n_cold"], r["cold_ttft_p95"],
                 "ABORT" if r["aborted"] else ""))
    print("summary.csv written:", os.path.join(root, "summary.csv"))


if __name__ == "__main__":
    main()
