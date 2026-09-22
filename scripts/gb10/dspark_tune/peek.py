import sys, json

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
print(len(rows), "requests")
for r in rows:
    u = r.get("usage", {}) or {}
    d = u.get("prompt_tokens_details", {}) or {}
    ttft = (r["ts_first"] - r["ts_sent"]) if r.get("ts_first") else None
    print(r.get("kind"), r.get("slot"), r.get("status"),
          ("ttft=%.3f" % ttft) if ttft is not None else "ttft=None",
          "pt=%s ct=%s cache=%s" % (u.get("prompt_tokens"),
                                    u.get("completion_tokens"),
                                    d.get("cached_tokens")))
