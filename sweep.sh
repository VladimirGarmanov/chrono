#!/usr/bin/env bash
#
# Parameter sweep for the Chrono::FSI SPH dam-break demo.
#
# Runs the demo once per (viscosity, fluid column height) pair, each into its own
# output directory, and appends one row per run to runs.csv.
#
# Usage:
#   ./sweep.sh            run the sweep
#   ./sweep.sh --dry-run  print the commands without running anything

set -u
set -o pipefail

# ---------------------------------------------------------------- configuration

DEMO="./build/bin/demo_FSI-SPH_DamBreak"
OUT_ROOT="DEMO_OUTPUT/FSI_Dam_Break"
MANIFEST="runs.csv"

# Swept parameters
VISCOSITIES=(1 2 5 10 20)
# Capped at 5.0: a column of height h reaches sqrt(2*g*h) in free fall, and
# past max_velocity below the weakly-compressible model blows up (density NaN).
# 6.0 and 7.0 give 10.8 and 11.7 m/s against a 10.0 ceiling and always abort.
HEIGHTS=(1.0 2.0 3.0 4.0 5.0)

# Held fixed across the sweep: the neighbourhood statistics the network sees
# depend on the particle spacing, so mixing resolutions in one dataset would
# make the training signal inconsistent.
SPACING=0.1
T_END=10
OUTPUT_FPS=200

# Abort before the root filesystem fills up (GiB). Each run writes ~3 GB.
MIN_FREE_GB=15

# ---------------------------------------------------------------------- colours

if [ -t 1 ]; then
    R=$'\e[0m'; B=$'\e[1m'; DIM=$'\e[2m'
    CY=$'\e[36m'; GR=$'\e[32m'; YE=$'\e[33m'; RD=$'\e[31m'; MG=$'\e[35m'
else
    R=''; B=''; DIM=''; CY=''; GR=''; YE=''; RD=''; MG=''
fi

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

TOTAL=$(( ${#VISCOSITIES[@]} * ${#HEIGHTS[@]} ))

banner() {
    printf '%s' "$CY"
    cat <<'ART'
   ___  _  _ ___ ___  _  _  ___     ___ _    _ ___ ___ ___
  / __|| || | _ \ _ \| \| |/ _ \   / __| |  | | _ \ __| _ \
 | (__ | __ | v / v /| .` | (_) |  \__ \ |/\| |   / _||  _/
  \___||_||_|_|_\_|_\|_|\_|\___/   |___/__/\__|_|_\___|_|
ART
    printf '%s' "$R"
    printf '%s ─── %d runs ─── %s × %s grid ───%s\n\n' \
        "$DIM" "$TOTAL" "${#VISCOSITIES[@]} visc" "${#HEIGHTS[@]} height" "$R"
}

# POSIX df: column 4 of the last line is the available space in 1K blocks.
free_gb() { df -k / | tail -1 | awk '{print int($4/1024/1024)}'; }

hms() { printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

bar() {  # bar <done> <total>
    local done=$1 total=$2 width=32
    local filled=$(( done * width / total ))
    printf '%s[%s' "$MG" "$GR"
    printf '%0.s█' $(seq 1 $filled) 2>/dev/null
    printf '%s' "$DIM"
    [ $filled -lt $width ] && printf '%0.s·' $(seq 1 $((width - filled)))
    printf '%s] %s%d/%d%s' "$MG" "$B" "$done" "$total" "$R"
}

# ------------------------------------------------------------------ preflight

banner

if [ ! -x "$DEMO" ] && [ "$DRY_RUN" = 0 ]; then
    printf '%s[!]%s demo binary not found: %s\n' "$RD" "$R" "$DEMO"
    printf '    build it first, or fix DEMO= at the top of this script\n'
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "run_tag,viscosity,fzDim,initial_spacing,t_end,output_fps,n_particles,wall_seconds,status" > "$MANIFEST"
fi

printf '%s[i]%s free disk: %s GiB   floor: %s GiB\n\n' "$CY" "$R" "$(free_gb)" "$MIN_FREE_GB"

# ---------------------------------------------------------------------- sweep

i=0
started=$(date +%s)

for v in "${VISCOSITIES[@]}"; do
    for h in "${HEIGHTS[@]}"; do
        i=$((i + 1))
        tag="v${v}_h${h}"
        run_dir="${OUT_ROOT}/${tag}"

        cmd=("$DEMO"
             --no_vis --output
             --t_end "$T_END"
             --output_fps "$OUTPUT_FPS"
             --initial_spacing "$SPACING"
             --viscosity "$v"
             --fzDim "$h"
             --run_tag "$tag")

        if [ "$DRY_RUN" = 1 ]; then
            printf '%s[%02d/%02d]%s %s\n' "$DIM" "$i" "$TOTAL" "$R" "${cmd[*]}"
            continue
        fi

        # Skip work that is already on disk, so an interrupted sweep can resume.
        if [ -d "${run_dir}/particles" ]; then
            printf '%s[%02d/%02d]%s %-14s %sskip (exists)%s\n' \
                "$DIM" "$i" "$TOTAL" "$R" "$tag" "$YE" "$R"
            continue
        fi

        avail=$(free_gb)
        if [ "$avail" -lt "$MIN_FREE_GB" ]; then
            printf '\n%s[!]%s only %s GiB left, stopping at run %d/%d\n' \
                "$RD" "$R" "$avail" "$i" "$TOTAL"
            break 2
        fi

        printf '%s[%02d/%02d]%s %sv=%-4s h=%-4s%s  ' \
            "$B$CY" "$i" "$TOTAL" "$R" "$B" "$v" "$h" "$R"

        t0=$(date +%s)
        log="/tmp/sweep_${tag}.log"
        if "${cmd[@]}" > "$log" 2>&1; then
            status=ok
        else
            status=FAILED
        fi
        t1=$(date +%s)
        wall=$((t1 - t0))

        # Particle count from the first output frame (minus the CSV header).
        first="${run_dir}/particles/fluid0.csv"
        if [ -f "$first" ]; then
            n_particles=$(( $(wc -l < "$first") - 1 ))
        else
            n_particles=0
        fi

        if [ "$status" = ok ]; then
            printf '%s✓%s %s  %s%d particles%s\n' \
                "$GR" "$R" "$(hms $wall)" "$DIM" "$n_particles" "$R"
        else
            printf '%s✗ failed%s  see %s\n' "$RD" "$R" "$log"
        fi

        echo "${tag},${v},${h},${SPACING},${T_END},${OUTPUT_FPS},${n_particles},${wall},${status}" >> "$MANIFEST"

        elapsed=$(( t1 - started ))
        printf '          '; bar "$i" "$TOTAL"
        printf '  %selapsed %s · free %s GiB%s\n\n' "$DIM" "$(hms $elapsed)" "$(free_gb)" "$R"
    done
done

# --------------------------------------------------------------------- summary

if [ "$DRY_RUN" = 1 ]; then
    printf '\n%s[i]%s dry run — nothing executed\n' "$CY" "$R"
    exit 0
fi

total_elapsed=$(( $(date +%s) - started ))
ok_count=$(grep -c ',ok$' "$MANIFEST" 2>/dev/null || echo 0)

printf '%s────────────────────────────────────────────────%s\n' "$DIM" "$R"
printf '%s[✓]%s done in %s · %s runs ok · manifest: %s%s%s\n' \
    "$GR" "$R" "$(hms $total_elapsed)" "$ok_count" "$B" "$MANIFEST" "$R"
printf '    disk now: %s GiB free\n' "$(free_gb)"
