import sys, json

d = json.load(sys.stdin)
res = d.get("data", {}).get("result", [])
for s in res:
    vals = s.get("values", [])
    out = []
    for t, v in vals:
        # маркируем сутки МСК: точка в 21:00 UTC кончает очередной день
        import datetime
        dt = datetime.datetime.utcfromtimestamp(float(t)) - datetime.timedelta(hours=3)
        day = dt.strftime("%d")
        out.append((day, round(float(v), 1)))
    m = "{" + ",".join("%s=%s" % (k, v) for k, v in s.get("metric", {}).items()) + "}"
    print(m + " " + " ".join("%s:%s" % (a, b) for a, b in out))
