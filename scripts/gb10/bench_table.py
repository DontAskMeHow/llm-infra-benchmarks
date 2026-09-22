#!/usr/bin/env python3
"""Render model comparison tables from benchmark result JSONs (standard v2).

Usage:
    python bench_table.py result_a.json result_b.json [result_c.json ...] [--base a]
    python bench_table.py results_dir/            # picks latest actual per model+hardware
    python bench_table.py a.json b.json --all     # include superseded results
    python bench_table.py --retire old.json new.json "retest reason"

Output: borderless aligned table (see benchmark/results/README.md for schema).
Stdlib only.
"""

import argparse
import json
import sys
from pathlib import Path

# (result key, row label)
METRIC_ROWS = [
    ("parameters", "parameters"),
    ("weight_size", "weight size"),
    ("cold_prefill_100k", "cold prefill 100K"),
    ("warm_prefill_100k", "warm prefill 100K"),
    ("decode_long_100k_c1", "decode long @100K c=1"),
    ("decode_short_4k_c1", "decode short @4K c=1"),
    ("kv_cache_gpu", "KV cache GPU"),
    ("kv_cache_cpu", "KV cache CPU"),
    ("cpu_offload", "cpu offload verdict"),
    ("spec_rate", "spec acceptance rate"),
    ("spec_length", "spec acceptance length"),
]

GAP = "  "


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_int(n):
    return f"{n:,.0f}"


def fmt_tokens(n):
    if n is None:
        return None
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}M" if v < 100 else f"{v:.0f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}K" if v < 10 else f"{v:.0f}K"
    return str(n)


def fmt_ms(ms):
    if ms is None:
        return None
    if ms >= 1000:
        s = ms / 1000
        return f"{s:.2f}s" if s < 10 else f"{s:.1f}s"
    return f"{ms:.0f}ms"


def fmt_pct(delta):
    """delta in percent, rounded to int; sign prefix."""
    if delta is None:
        return None
    sign = "+" if delta >= 0 else "\u2212"
    return f"{sign}{abs(round(delta))}%"


def pct_change(base, new):
    if base is None or new is None or base == 0:
        return None
    return (new - base) / base * 100.0


def get_spec_method(r):
    spec = r.get("environment", {}).get("spec_decoding") or {}
    method = spec.get("method")
    n = spec.get("num_speculative_tokens")
    if method and n:
        return f"{method}\u00d7{n}"
    return method or None


def cell(r, key):
    """Extract formatted cell text and comparable numeric base for a metric row."""
    res = r.get("results", {})
    m = r.get("model", {})

    if key == "parameters":
        p = m.get("parameters") or {}
        total, active = p.get("total_b"), p.get("active_b")
        if total is None:
            return None, None
        text = f"{fmt_b(total)} total"
        if active is not None:
            text += f", {fmt_b(active)} active"
        return text, None

    if key == "weight_size":
        gb, quant = m.get("weight_size_gb"), m.get("quant")
        if gb is None:
            return None, None
        return f"{fmt_int(gb)} GB ({quant})" if quant else f"{fmt_int(gb)} GB", None

    if key in ("cold_prefill_100k", "warm_prefill_100k"):
        d = res.get(key) or {}
        tok, ttft = d.get("tok_s"), d.get("ttft_ms")
        if tok is None:
            return None, None
        text = f"{fmt_int(tok)} tok/s"
        if ttft is not None:
            text += f" (TTFT {fmt_ms(ttft)})"
        return text, tok

    if key in ("decode_long_100k_c1", "decode_short_4k_c1"):
        d = res.get(key) or {}
        tok, tpot = d.get("tok_s"), d.get("tpot_ms")
        if tok is None:
            return None, None
        text = f"{tok:.1f} tok/s"
        if tpot is not None:
            text += f" (TPOT {tpot}ms)"
        return text, tok

    if key == "kv_cache_gpu":
        d = res.get("kv_cache_gpu") or {}
        tokens, gb = d.get("tokens"), d.get("gb_per_gpu")
        if tokens is None and gb is None:
            return None, None
        text = ""
        if tokens is not None:
            text += f"{fmt_tokens(tokens)} tokens"
        if gb is not None:
            text += f" ({gb} GB/GPU)"
        return text, (tokens, gb)

    if key == "kv_cache_cpu":
        d = res.get("kv_cache_cpu") or {}
        tokens, gb = d.get("tokens"), d.get("gb_total")
        if tokens is None and gb is None:
            return None, None
        parts = []
        if tokens is not None:
            parts.append(f"{fmt_tokens(tokens)} tokens")
        if gb is not None:
            parts.append(f"{fmt_int(gb)} GB")
        return " ".join(parts), tokens

    if key == "cpu_offload":
        d = res.get("kv_cache_cpu_restore") or {}
        verdict = d.get("verdict")
        if not verdict:
            off = r.get("environment", {}).get("kv_offload")
            if not off or not off.get("connector"):
                return "no_offload", None
            return None, None
        hits = d.get("external_hit_rate_pct")
        text = verdict
        if hits is not None:
            text += f" ({hits}% ext hits)"
        return text, None

    if key == "spec_rate":
        d = res.get("spec_acceptance") or {}
        rate = d.get("rate_pct")
        if rate is None:
            return None, None
        method = get_spec_method(r)
        if isinstance(rate, str):
            text = f"~{rate.lstrip('~')}%"
            num = None
            try:
                num = float(rate.lstrip("~"))
            except ValueError:
                pass
        else:
            text = f"{round(rate)}%"
            num = rate
        if method:
            text += f" ({method})"
        return text, num

    if key == "spec_length":
        d = res.get("spec_acceptance") or {}
        length = d.get("length_tokens")
        if length is None:
            return None, None
        if isinstance(length, str):
            return f"~{length.lstrip('~+')} tok/step", None
        return f"{length} tok/step", length

    return None, None


def fmt_b(n):
    """743 -> '743B', 1600 -> '1.6T'"""
    if n >= 1000:
        return f"{n / 1000:.1f}T"
    return f"{n}B"


def delta_cell(key, base_val, new_val):
    """Build delta column text for a metric row."""
    if key == "kv_cache_gpu":
        if not isinstance(base_val, tuple) or not isinstance(new_val, tuple):
            return None
        bt, bg = base_val
        nt, ng = new_val
        dt = fmt_pct(pct_change(bt, nt))
        dg = fmt_pct(pct_change(bg, ng))
        parts = []
        if dt:
            parts.append(f"{dt} tok")
        if dg:
            parts.append(f"{dg} GB")
        return ", ".join(parts) if parts else None

    if key in ("parameters", "weight_size", "cpu_offload"):
        return None  # not comparable as percent

    d = fmt_pct(pct_change(base_val, new_val))
    return d


def column_header(r):
    name = r.get("model", {}).get("name", "?")
    eng = r.get("engine", {})
    env = r.get("environment", {})
    framework = eng.get("framework", "?")
    version = eng.get("version", "?")
    date = env.get("date", "?")
    hw = env.get("hardware", "?")
    sup = " \u2020" if r.get("meta", {}).get("status") == "superseded" else ""
    return name + sup, f"{hw}, {framework} {version}, {date}"


def render(results, base_idx=0):
    headers = [column_header(r) for r in results]
    names = [h[0] for h in headers]
    details = [h[1] for h in headers]
    multi = len(results) > 1

    # rows: (label, cell_texts, delta_text)
    # Delta semantics: delta describes the FIRST column
    # relative to the reference column: (first - base) / base.
    # base = --base column, or the LAST column by default.
    table = []
    for key, label in METRIC_ROWS:
        cells, vals = [], []
        for r in results:
            text, val = cell(r, key)
            cells.append(text if text is not None else "\u2014")
            vals.append(val)
        d = None
        if multi:
            # subject = first column that is not the base
            subj_idx = 0 if base_idx != 0 else (1 if len(results) > 1 else 0)
            if subj_idx != base_idx:
                # delta = (subject - base) / base, e.g. GLM vs DeepSeek reference
                d = delta_cell(key, vals[base_idx], vals[subj_idx])
        table.append((label, cells, d))

    # widths
    label_w = max(len(label) for label, _, _ in table)
    col_w = []
    for i in range(len(results)):
        w = max(len(names[i]), len(details[i]),
                max(len(cells[i]) for _, cells, _ in table))
        col_w.append(w)
    delta_texts = [d for _, _, d in table if d]
    delta_w = max([len("\u0394")] + [len(d) for d in delta_texts]) if multi else 0

    # hardware line
    hws = {r.get("environment", {}).get("hardware") for r in results}
    title = f"test on {hws.pop()}" if len(hws) == 1 else "comparison (mixed hardware)"

    lines = [title, ""]
    hdr = "\u041c\u0435\u0442\u0440\u0438\u043a\u0430".ljust(label_w) + GAP + \
        GAP.join(n.ljust(col_w[i]) for i, n in enumerate(names))
    if multi:
        hdr += GAP + "\u0394".ljust(delta_w)
    lines.append(hdr.rstrip())
    sub = "".ljust(label_w) + GAP + \
        GAP.join(d.ljust(col_w[i]) for i, d in enumerate(details))
    lines.append(sub.rstrip())

    for label, cells, d in table:
        row = label.ljust(label_w) + GAP + \
            GAP.join(c.ljust(col_w[i]) for i, c in enumerate(cells))
        if multi:
            row += GAP + (d if d else "")
        lines.append(row.rstrip())

    lines.append("")

    # engine mismatch warning
    engines = {
        (r.get("engine", {}).get("framework"), r.get("engine", {}).get("version"))
        for r in results
    }
    if len(engines) > 1:
        eng_desc = ", ".join(f"{f} {v}" for f, v in engines)
        lines.append(f"warning: columns measured on different engines: {eng_desc}")

    return "\n".join(lines)


def pick_latest_actual(paths):
    """From a list of json paths, keep the latest actual result per (model, hardware)."""
    by_key = {}
    for p in paths:
        r = load(p)
        meta = r.get("meta", {})
        if meta.get("status", "actual") != "actual":
            continue
        key = (r.get("model", {}).get("name"),
               r.get("environment", {}).get("hardware"))
        date = r.get("environment", {}).get("date", "")
        if key not in by_key or date > by_key[key][0]:
            by_key[key] = (date, p)
    return [p for _, p in by_key.values()]


def collect(paths, show_all):
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    if not show_all:
        files = pick_latest_actual(files)
    else:
        # keep order, dedupe
        seen = set()
        uniq = []
        for f in files:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        files = uniq
    if len(files) < 1:
        sys.exit("no result files found")
    return files


def do_retire(old_path, new_path, reason):
    old = load(old_path)
    new = load(new_path)
    old.setdefault("meta", {})
    new.setdefault("meta", {})
    old["meta"]["status"] = "superseded"
    old["meta"]["superseded_by"] = Path(new_path).name
    new["meta"]["replaces"] = Path(old_path).name
    new["meta"]["retest_reason"] = reason
    # write both, old first
    for path, data in ((old_path, old), (new_path, new)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(f"retired {Path(old_path).name} -> superseded by {Path(new_path).name}")
    print(f"reason: {reason}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="result JSON files or a results directory")
    ap.add_argument("--base", default=None,
                    help="base column for delta: model name or file name (default: first)")
    ap.add_argument("--all", action="store_true",
                    help="include superseded results (marked with dagger)")
    ap.add_argument("--retire", nargs=3, metavar=("OLD", "NEW", "REASON"),
                    help="mark OLD superseded by NEW with retest reason")
    args = ap.parse_args()

    if args.retire:
        do_retire(*args.retire)
        return

    if not args.inputs:
        ap.error("provide result files or a directory")

    files = collect(args.inputs, args.all)
    results = [load(f) for f in files]

    # Default base for delta = LAST column (reference); delta describes the first.
    base_idx = len(results) - 1
    if args.base:
        base_idx = 0
        for i, r in enumerate(results):
            if r.get("model", {}).get("name") == args.base or \
                    Path(files[i]).name == args.base:
                base_idx = i
                break
        else:
            sys.exit(f"base not found: {args.base}")

    print(render(results, base_idx))


if __name__ == "__main__":
    main()
