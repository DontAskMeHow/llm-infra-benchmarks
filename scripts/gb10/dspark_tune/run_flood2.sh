#!/bin/sh
# Рабочий каталог с genload.py и outdir'ами (переопределяется DSTUNE_DIR).
DSTUNE_DIR="${DSTUNE_DIR:-/tmp/dspark-tune}"
cd "$DSTUNE_DIR" || exit 1
for N in 4 5 8 12; do
  echo "=== flood N=$N start $(date +%s) ==="
  python3 genload.py --phase t2 --n "$N" --seconds 100 --max-tokens 400 \
    --cold-rate 2 --cold-words 22000 --cold-max-tokens 32 \
    --abort-waiting 0 --outdir "T4-flood/f-$(printf %02d "$N")" \
    --run-id "f-$(printf %02d "$N")" || echo "FAIL N=$N"
  echo "=== flood N=$N end $(date +%s) ==="
  sleep 6
done
echo ALLDONE
