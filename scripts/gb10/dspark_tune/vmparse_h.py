import sys, json, datetime

d = json.load(sys.stdin)
res = d.get("data", {}).get("result", [])
for s in res:
    vals = s.get("values", [])
    out = []
    for t, v in vals:
        dt = datetime.datetime.fromtimestamp(float(t), datetime.UTC)
        msk = dt - datetime.timedelta(hours=3)
        out.append((msk.strftime("%m%d.%H"), round(float(v), 1)))
    m = "{" + ",".join("%s=%s" % (k, v) for k, v in s.get("metric", {}).items()) + "}"
    print(m + " " + " ".join("%s:%s" % (a, b) for a, b in out))
