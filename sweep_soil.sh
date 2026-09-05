#!/usr/bin/env bash
#
# Parameter sweep for the Chrono::FSI CRM plate-sinkage demo.
#
# Runs the demo once per (friction angle, cohesion) pair and appends one row per
# run to runs_soil.csv.
#
# The demo has no --run_tag: its output directory is fixed (out_dir in
# demo_FSI-SPH_PlateSinkage.cpp:413), so every run would write into the same
# place. Each run therefore gets a clean directory before it starts and has its
# output moved aside afterwards.
#
# Usage:
#   ./sweep_soil.sh            run the sweep
#   ./sweep_soil.sh --dry-run  print the commands without running anything

set -u
set -o pipefail

# ---------------------------------------------------------------- configuration

# Build directory: override with DEMO=... , otherwise prefer the ninja build
# when one exists. The choice is printed rather than silently taken, so a stale
# binary in the other tree cannot be used without it showing up in the log.
if [ -z "${DEMO:-}" ]; then
    for candidate in ./build-ninja/bin/demo_FSI-SPH_PlateSinkage \
                     ./build/bin/demo_FSI-SPH_PlateSinkage; do
        [ -x "$candidate" ] && { DEMO="$candidate"; break; }
    done
    DEMO="${DEMO:-./build/bin/demo_FSI-SPH_PlateSinkage}"
fi
WORK_DIR="DEMO_OUTPUT/FSI_Plate_Sinkage"   # fixed by the demo
RUNS_DIR="DEMO_OUTPUT/plate_runs"          # where finished runs are parked
MANIFEST="runs_soil.csv"

# Swept parameters: the two that dominate bearing capacity.
PHI_DEGS=(25 30 35 40 45)        # internal friction angle [deg]
COHESIONS=(0 500 1000 2000 5000) # cohesion [Pa]

# Held fixed. Particle spacing governs the neighbourhood statistics the network
# sees, so it must not vary within one dataset. 0.02 is the demo's documented
# coarse preview (~5 min/run) rather than the 0.0125 converged default, which
# would put this sweep past both the disk and the time budget.
SPACING=0.02
PARTICLE_FPS=50
RHEOLOGY=mu_i

# Abort before the root filesystem fills up (GiB).
MIN_FREE_GB=15

# ---------------------------------------------------------------------- colours

if [ -t 1 ]; then
    R=$'\e[0m'; B=$'\e[1m'; DIM=$'\e[2m'
    CY=$'\e[36m'; GR=$'\e[32m'; YE=$'\e[33m'; RD=$'\e[31m'
else
    R=''; B=''; DIM=''; CY=''; GR=''; YE=''; RD=''
fi

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

TOTAL=$(( ${#PHI_DEGS[@]} * ${#COHESIONS[@]} ))

free_gb() { df -k / | tail -1 | awk '{print int($4/1024/1024)}'; }
hms() { printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

bar() {  # bar <done> <total>
    local done=$1 total=$2 width=32
    local filled=$(( done * width / total ))
    printf '%s[%s' "$DIM" "$GR"
    [ $filled -gt 0 ] && printf '%0.s#' $(seq 1 $filled)
    printf '%s' "$DIM"
    [ $filled -lt $width ] && printf '%0.s.' $(seq 1 $((width - filled)))
    printf ']%s %s%d/%d%s' "$DIM" "$B" "$done" "$total" "$R"
}

# A run counts as done only if the manifest says it finished, not merely because
# a directory exists: an interrupted run leaves a partial directory behind.
already_done() {
    [ -f "$MANIFEST" ] && grep -q "^$1,.*,ok$" "$MANIFEST"
}

# ------------------------------------------------------------------ preflight

printf '%s CRM plate-sinkage sweep %s\n' "$B$CY" "$R"
printf '%s %d runs · %d phi × %d cohesion · spacing %s · %s fps%s\n' \
    "$DIM" "$TOTAL" "${#PHI_DEGS[@]}" "${#COHESIONS[@]}" "$SPACING" "$PARTICLE_FPS" "$R"
printf '%s binary: %s%s\n\n' "$DIM" "$DEMO" "$R"

if [ ! -x "$DEMO" ] && [ "$DRY_RUN" = 0 ]; then
    printf '%s[!]%s demo binary not found: %s\n' "$RD" "$R" "$DEMO"
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "run_tag,phi_deg,cohesion,spacing,particle_fps,rheology,n_particles,n_frames,wall_seconds,status" > "$MANIFEST"
fi

mkdir -p "$RUNS_DIR"
printf '%s[i]%s free disk: %s GiB   floor: %s GiB\n\n' "$CY" "$R" "$(free_gb)" "$MIN_FREE_GB"

# ---------------------------------------------------------------------- sweep

i=0
started=$(date +%s)

for phi in "${PHI_DEGS[@]}"; do
    for coh in "${COHESIONS[@]}"; do
        i=$((i + 1))
        tag="phi${phi}_c${coh}"

        cmd=("$DEMO"
             --no_vis
             --quiet
             --rheology "$RHEOLOGY"
             --spacing "$SPACING"
             --particle_fps "$PARTICLE_FPS"
             --phi_deg "$phi"
             --cohesion "$coh")

        if [ "$DRY_RUN" = 1 ]; then
            printf '%s[%02d/%02d]%s %s\n' "$DIM" "$i" "$TOTAL" "$R" "${cmd[*]}"
            continue
        fi

        if already_done "$tag"; then
            printf '%s[%02d/%02d]%s %-16s %sskip (done)%s\n' \
                "$DIM" "$i" "$TOTAL" "$R" "$tag" "$YE" "$R"
            continue
        fi

        avail=$(free_gb)
        if [ "$avail" -lt "$MIN_FREE_GB" ]; then
            printf '\n%s[!]%s only %s GiB left, stopping at run %d/%d\n' \
                "$RD" "$R" "$avail" "$i" "$TOTAL"
            break 2
        fi

        printf '%s[%02d/%02d]%s %sphi=%-3s c=%-5s%s  ' \
            "$B$CY" "$i" "$TOTAL" "$R" "$B" "$phi" "$coh" "$R"

        # Clean slate: the demo always writes to the same fixed directory, and
        # leftovers from a crashed run would be mixed into this one's output.
        rm -rf "$WORK_DIR" "${RUNS_DIR:?}/$tag"

        t0=$(date +%s)
        log="/tmp/sweep_soil_${tag}.log"
        if "${cmd[@]}" > "$log" 2>&1; then
            status=ok
        else
            status=FAILED
        fi
        wall=$(( $(date +%s) - t0 ))

        n_particles=0
        n_frames=0
        if [ -d "$WORK_DIR/particles" ]; then
            n_frames=$(find "$WORK_DIR/particles" -name 'fluid*.csv' | wc -l | tr -d ' ')
            first="$WORK_DIR/particles/fluid0.csv"
            [ -f "$first" ] && n_particles=$(( $(wc -l < "$first") - 1 ))
            mv "$WORK_DIR" "$RUNS_DIR/$tag"
        else
            status=FAILED
        fi

        if [ "$status" = ok ]; then
            printf '%s✓%s %s  %s%d particles · %d frames%s\n' \
                "$GR" "$R" "$(hms $wall)" "$DIM" "$n_particles" "$n_frames" "$R"
        else
            printf '%s✗ failed%s  see %s\n' "$RD" "$R" "$log"
        fi

        echo "${tag},${phi},${coh},${SPACING},${PARTICLE_FPS},${RHEOLOGY},${n_particles},${n_frames},${wall},${status}" >> "$MANIFEST"

        printf '          '; bar "$i" "$TOTAL"
        printf '  %selapsed %s · free %s GiB%s\n\n' \
            "$DIM" "$(hms $(( $(date +%s) - started )))" "$(free_gb)" "$R"
    done
done

# --------------------------------------------------------------------- summary

if [ "$DRY_RUN" = 1 ]; then
    printf '\n%s[i]%s dry run — nothing executed\n' "$CY" "$R"
    exit 0
fi

ok_count=$(grep -c ',ok$' "$MANIFEST" 2>/dev/null || echo 0)
printf '%s[done]%s %s · %s runs ok · manifest: %s · free %s GiB\n' \
    "$GR" "$R" "$(hms $(( $(date +%s) - started )))" "$ok_count" "$MANIFEST" "$(free_gb)"
