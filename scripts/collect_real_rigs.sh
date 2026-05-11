#!/bin/bash
# Collect CARLA data using every rig in `rigs/real_rigs/` (one CARLA run, all
# rigs see the same scenes). Two preset modes:
#
#   scripts/collect_real_rigs.sh test     # 2 routes × 90 s, fast inspection
#   scripts/collect_real_rigs.sh full     # 50 routes × 10 min, production
#
# Output goes to `data/real_rigs_<mode>/<rig-name>/logs/...` so each rig can
# be loaded independently with viser or the analyzer.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-test}"

case "$MODE" in
  test)
    MAX_ROUTES=2
    TIMEOUT=90
    OUTPUT="data/real_rigs_test"
    ;;
  full)
    MAX_ROUTES=50
    TIMEOUT=600
    OUTPUT="data/real_rigs"
    ;;
  *)
    echo "Usage: $0 {test|full}" >&2
    exit 1
    ;;
esac

# Use Town05 LEAD-curated routes — the 50x38_Town12 routes spawn the ego at
# z=370 and physics-drops it through the world (see prior debugging session).
ROUTES_DIR="data/data_routes/lead/Accident"

export PYTHONPATH="3rd_party/CARLA_0915/PythonAPI/carla:3rd_party/leaderboard_autopilot:3rd_party/scenario_runner_autopilot:${PYTHONPATH:-}"

echo "[collect_real_rigs] mode=${MODE}  routes=${MAX_ROUTES}  timeout=${TIMEOUT}s  output=${OUTPUT}"
rm -f data/multi_rig_debug/results/*.json

exec /home/bastian/dev/master-thesis/.venv_training/bin/python scripts/collect_carla_data.py \
  --rigs-dir rigs/real_rigs \
  --routes-dir "$ROUTES_DIR" \
  --max-routes "$MAX_ROUTES" \
  --routes-seed 0 \
  --per-route-timeout "$TIMEOUT" \
  --output-base "$OUTPUT"
