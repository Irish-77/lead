#!/bin/bash
# Full-dataset collection: 6 real rigs + 16 random rigs, target ≥ 5000
# saved iters per rig.
#
# Designed to run in the background and survive disconnect:
#
#   nohup scripts/collect_full_dataset.sh > /tmp/full_dataset.log 2>&1 &
#   disown
#
# CARLA is restarted between every batch of ROUTES_PER_CARLA routes — without
# this, CARLA accumulates memory and gets OOM-killed after ~20 minutes when
# many rigs / sensors are active. We saw this on the previous full-dataset
# attempt (CARLA RSS hit ~17 GB on a 31 GB host and was killed by oomd at
# minute 23, after which the orchestrator silently kept attempting routes
# against a dead server for 9 hours).
#
# Resumable: if interrupted, just re-run — already-collected frames count
# toward the target, so each phase only runs additional cycles as needed.
#
# Splits 22 rigs into 3 phases (real / random_a / random_b) so concurrent
# camera count stays in the range we know the GPU handles cleanly.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGET_FRAMES=${TARGET_FRAMES:-6000}
MIN_ITERS_PER_LOG=${MIN_ITERS_PER_LOG:-10}
ROUTES_DIR=${ROUTES_DIR:-data/data_routes/lead_smalltowns}
PER_ROUTE_TIMEOUT=${PER_ROUTE_TIMEOUT:-300}
ROUTES_PER_CARLA=${ROUTES_PER_CARLA:-3}
CYCLES_PER_PHASE=${CYCLES_PER_PHASE:-200}
CARLA_PORT=${CARLA_PORT:-2000}
CARLA_TM_PORT=${CARLA_TM_PORT:-8000}
PYBIN="/home/bastian/dev/master-thesis/.venv_training/bin/python"
CARLA_SH="3rd_party/CARLA_0915/CarlaUE4.sh"

export PYTHONPATH="3rd_party/CARLA_0915/PythonAPI/carla:3rd_party/leaderboard_autopilot:3rd_party/scenario_runner_autopilot:${PYTHONPATH:-}"

# (phase_name, rigs_dir, output_dir)
# Synthetic (random) phases run first because they had no logs yet and to
# exercise the new aspect-ratio dimension jitter; real phase resumes its
# already-saved data afterward.
PHASES=(
    "random_a:rigs/random_batch_a:data/dataset_full/phase_random_a"
    "random_b:rigs/random_batch_b:data/dataset_full/phase_random_b"
    "real:rigs/real_rigs:data/dataset_full/phase_real"
)

stop_carla() {
    # Multiple match patterns because CARLA's actual binary name is
    # "CarlaUE4-Linux-Shipping" (the suffix matters; a previous version of
    # this function used the wrong substring and left stale CARLAs behind,
    # holding port 2000 and breaking every subsequent cycle).
    # Also matches `python ... collect_carla_data.py` so an orphaned
    # collect_carla_data process from a prior wrapper death gets cleaned up.
    pkill -INT -f "leaderboard_evaluator_local" 2>/dev/null || true
    pkill -INT -f "collect_carla_data.py"        2>/dev/null || true
    pkill -INT -f "CarlaUE4.sh"                  2>/dev/null || true
    pkill -INT -f "CarlaUE4-Linux-Shipping"      2>/dev/null || true
    pkill -INT -f "CarlaUE4-Linux"               2>/dev/null || true
    pkill -INT -f "CarlaUnreal"                  2>/dev/null || true
    sleep 5
    pkill -KILL -f "leaderboard_evaluator_local" 2>/dev/null || true
    pkill -KILL -f "collect_carla_data.py"       2>/dev/null || true
    pkill -KILL -f "CarlaUE4.sh"                 2>/dev/null || true
    pkill -KILL -f "CarlaUE4-Linux-Shipping"     2>/dev/null || true
    pkill -KILL -f "CarlaUE4-Linux"              2>/dev/null || true
    pkill -KILL -f "CarlaUnreal"                 2>/dev/null || true
    sleep 2

    # Wait for the rpc port to actually free; without this the new CARLA
    # tries to bind, fails ("Address already in use"), and segfaults.
    local waited=0
    while ss -tlnp 2>/dev/null | grep -q ":${CARLA_PORT}\b"; do
        sleep 2
        waited=$((waited + 2))
        if [ ${waited} -ge 30 ]; then
            echo "[$(date)] WARN: port ${CARLA_PORT} still bound after 30s"
            break
        fi
    done
}

# If the wrapper dies for any reason (kill, OOM, host shutdown), make sure
# we don't leave orphaned CARLA / leaderboard processes spinning. Without
# this, repeated relaunches accumulate stale CARLAs that fight over port 2000.
trap 'echo "[$(date)] wrapper exiting; cleaning up children"; stop_carla' EXIT INT TERM

start_carla() {
    # Default quality. -quality-level=Low was tested but caused some
    # sensors to never deliver data (`SensorReceivedNoData` from the
    # leaderboard agent_wrapper), so it's not a viable memory mitigation.
    nohup "${CARLA_SH}" -RenderOffScreen -nosound -graphicsadapter=0 \
        -carla-rpc-port="${CARLA_PORT}" \
        > /tmp/carla_server.log 2>&1 &
    disown
    # Wait up to 90 s for the RPC port to come up.
    local waited=0
    while ! ss -tlnp 2>/dev/null | grep -q ":${CARLA_PORT}"; do
        sleep 2
        waited=$((waited + 2))
        if [ ${waited} -ge 90 ]; then
            echo "[$(date)] CARLA failed to come up within 90s"
            return 1
        fi
    done
    echo "[$(date)] CARLA listening on :${CARLA_PORT}"
    sleep 3   # let the world settle
}

run_chunk() {
    # One CARLA invocation worth of routes. Caller already started CARLA.
    local rigs_dir=$1 output=$2 cycle=$3 chunk=$4

    # cycle * something + chunk gives a unique seed per chunk; collect_carla_data
    # uses the seed to shuffle the route list, so different chunks see different
    # routes (and different traffic).
    local seed=$((cycle * 1000 + chunk))

    rm -f data/multi_rig_debug/results/*.json 2>/dev/null

    "${PYBIN}" scripts/collect_carla_data.py \
        --rigs-dir "${rigs_dir}" \
        --routes-dir "${ROUTES_DIR}" \
        --max-routes "${ROUTES_PER_CARLA}" \
        --routes-seed "${seed}" \
        --traffic-manager-seed "${seed}" \
        --carla-port "${CARLA_PORT}" \
        --traffic-manager-port "${CARLA_TM_PORT}" \
        --per-route-timeout "${PER_ROUTE_TIMEOUT}" \
        --output-base "${output}" 2>&1 || \
        echo "[$(date)] orchestrator exit nonzero — continuing"
}

run_phase() {
    local name=$1 rigs_dir=$2 output=$3

    echo ""
    echo "================================================================"
    echo "[$(date)] Phase ${name}: ${rigs_dir} -> ${output}"
    echo "================================================================"

    for cycle in $(seq 0 $((CYCLES_PER_PHASE - 1))); do
        # Done?
        if "${PYBIN}" scripts/dataset_progress.py "${output}" --target "${TARGET_FRAMES}" 2>/dev/null; then
            echo "[$(date)] Phase ${name}: target reached, skipping further cycles"
            return 0
        fi

        echo ""
        echo "[$(date)] Phase ${name} cycle ${cycle}/${CYCLES_PER_PHASE}"

        stop_carla
        if ! start_carla; then
            echo "[$(date)] Could not start CARLA, skipping cycle"
            sleep 30
            continue
        fi

        run_chunk "${rigs_dir}" "${output}" "${cycle}" 0

        # Prune short logs from this cycle (and any prior). Routes that went
        # immediately bad leave <10-iter logs that pollute training data; we
        # delete them so the target-frames check only counts useful frames.
        "${PYBIN}" scripts/prune_short_logs.py "${output}" \
            --min-iters "${MIN_ITERS_PER_LOG}" 2>&1 | tail -5 || true

        echo "[$(date)] Phase ${name} cycle ${cycle} progress (after prune):"
        "${PYBIN}" scripts/dataset_progress.py "${output}" 2>/dev/null || true
    done

    echo ""
    echo "[$(date)] Phase ${name}: ran ${CYCLES_PER_PHASE} cycles, final progress:"
    "${PYBIN}" scripts/dataset_progress.py "${output}" --target "${TARGET_FRAMES}" || true
}

echo "[$(date)] === Full dataset collection started ==="
echo "  TARGET_FRAMES=${TARGET_FRAMES}"
echo "  MIN_ITERS_PER_LOG=${MIN_ITERS_PER_LOG}"
echo "  ROUTES_PER_CARLA=${ROUTES_PER_CARLA}"
echo "  PER_ROUTE_TIMEOUT=${PER_ROUTE_TIMEOUT}"
echo "  CYCLES_PER_PHASE=${CYCLES_PER_PHASE}"
echo "  ROUTES_DIR=${ROUTES_DIR}"

for entry in "${PHASES[@]}"; do
    IFS=":" read -r name rigs_dir output <<< "${entry}"
    run_phase "${name}" "${rigs_dir}" "${output}"
done

stop_carla

echo ""
echo "[$(date)] === Full dataset collection finished ==="
echo ""
echo "Final tally:"
for entry in "${PHASES[@]}"; do
    IFS=":" read -r name rigs_dir output <<< "${entry}"
    echo ""
    echo "Phase ${name}:"
    "${PYBIN}" scripts/dataset_progress.py "${output}" --target "${TARGET_FRAMES}" 2>/dev/null || true
done
