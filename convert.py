#!/usr/bin/env python3
"""Pack the per-frame CSV output of a Chrono::FSI SPH sweep into one array per run.

Chrono writes one CSV per output frame, which is both bulky (text) and slow to
read at training time. This collapses each run into a single float32 array of
shape (n_frames, n_particles, n_features).

Both physics problems are handled; the format is detected from each run's CSV
header, since the two solvers write different columns (SphUtilsPrint.cu).

    CFD (water)   8 features:  x y z  v_x v_y v_z  rho pressure
    CRM (soil)   16 features:  x y z  v_x v_y v_z  rho
                               p11 p22 p33  shear12 shear13 shear23
                               pc Ev Sv

Dropped in both cases: |U| and acc are magnitudes derivable from the velocity
and force columns, and CRM's h is the smoothing length, constant within a run.

Usage:
    ./convert.py --root DEMO_OUTPUT/plate_runs --out dataset_soil
    ./convert.py --root DEMO_OUTPUT/plate_runs --out dataset_soil --delete
"""

import argparse
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required (pip install pandas) — it reads these CSVs "
             "roughly 15x faster than numpy's own parsers")

# Columns to keep, in output order, per physics problem. The CRM names carry the
# solver's own field names in parentheses; they are matched verbatim against the
# header Chrono writes.
KEEP_CFD = ["x", "y", "z", "v_x", "v_y", "v_z", "rho", "pressure"]

KEEP_CRM = [
    "x", "y", "z",
    "v_x", "v_y", "v_z",
    "rho(rpx)",
    "p11(tauXxYyZz_11)", "p22(tauXxYyZz_22)", "p33(tauXxYyZz_33)",
    "shear12(tauXyXzYz_12)", "shear13(tauXyXzYz_13)", "shear23(tauXyXzYz_23)",
    "pc", "Ev", "Sv",
]

# Chrono numbers its frames without zero padding, so fluid10 sorts before
# fluid2 lexically. Ordering by the embedded integer is what keeps the frames
# in time order — getting this wrong silently scrambles every trajectory.
FRAME_RE = re.compile(r"(\d+)\.csv$")


def frame_index(path):
    m = FRAME_RE.search(path.name)
    if m is None:
        raise ValueError(f"cannot read a frame number from {path.name}")
    return int(m.group(1))


def detect_columns(path):
    """Return (label, columns) for a run, from the header of one of its frames."""
    header = pd.read_csv(path, skipinitialspace=True, nrows=0).columns.tolist()
    for label, keep in (("CRM", KEEP_CRM), ("CFD", KEEP_CFD)):
        if all(c in header for c in keep):
            return label, keep
    raise ValueError(f"unrecognised CSV header in {path.name}: {header}")


def read_frame(args):
    path, keep = args
    df = pd.read_csv(path, skipinitialspace=True)
    return df[keep].to_numpy(dtype=np.float32)


def convert_run(run_dir, out_dir, jobs, delete):
    """Pack one run. Returns (error, label, n_features); error is None on success."""
    particles = run_dir / "particles"
    frames = sorted(particles.glob("fluid*.csv"), key=frame_index)
    if not frames:
        return "no fluid*.csv frames", None, 0

    # A gap in the numbering means the run died partway through and the frames
    # are not a contiguous trajectory.
    numbers = [frame_index(f) for f in frames]
    if numbers != list(range(len(numbers))):
        return (f"frame numbering is not contiguous "
                f"(got {numbers[0]}..{numbers[-1]}, {len(numbers)} files)"), None, 0

    label, keep = detect_columns(frames[0])
    first = read_frame((frames[0], keep))
    n_particles, n_cols = first.shape
    data = np.empty((len(frames), n_particles, n_cols), dtype=np.float32)
    data[0] = first

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        work = ((f, keep) for f in frames[1:])
        for i, arr in enumerate(pool.map(read_frame, work, chunksize=8), start=1):
            if arr.shape[0] != n_particles:
                return f"frame {i} has {arr.shape[0]} particles, expected {n_particles}", label, n_cols
            data[i] = arr

    if not np.isfinite(data).all():
        return "array contains NaN or inf — the run diverged", label, n_cols

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{run_dir.name}.npy", data)

    # The boundary markers are static, so one frame of them is enough.
    boundary = particles / "boundary0.csv"
    if boundary.exists():
        np.save(out_dir / f"{run_dir.name}_boundary.npy", read_frame((boundary, keep)))

    if delete:
        shutil.rmtree(particles)

    return None, label, n_cols


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}PB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="DEMO_OUTPUT/plate_runs",
                    help="directory holding the per-run subdirectories")
    ap.add_argument("--out", default="dataset_soil", help="where to write the .npy files")
    ap.add_argument("--jobs", type=int, default=8, help="parallel CSV readers")
    ap.add_argument("--only", default=None,
                    help="comma-separated run tags to pack, instead of every run found")
    ap.add_argument("--delete", action="store_true",
                    help="remove each run's particles/ directory once it is packed")
    args = ap.parse_args()

    root, out_dir = Path(args.root), Path(args.out)
    runs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "particles").is_dir())
    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        runs = [d for d in runs if d.name in wanted]
        missing = wanted - {d.name for d in runs}
        if missing:
            sys.exit(f"no packable run directory for: {', '.join(sorted(missing))}")
    if not runs:
        sys.exit(f"no runs with a particles/ directory under {root}")

    print(f"{len(runs)} runs under {root}\n")
    packed = skipped = 0
    total_out = 0
    started = time.time()

    for i, run_dir in enumerate(runs, start=1):
        print(f"[{i:02d}/{len(runs)}] {run_dir.name:<14} ", end="", flush=True)
        t0 = time.time()
        csv_bytes = sum(f.stat().st_size for f in (run_dir / "particles").glob("*.csv"))

        error, label, n_cols = convert_run(run_dir, out_dir, args.jobs, args.delete)
        if error:
            print(f"skip — {error}")
            skipped += 1
            continue

        npy_bytes = (out_dir / f"{run_dir.name}.npy").stat().st_size
        total_out += npy_bytes
        packed += 1
        print(f"{label} x{n_cols}  {human(csv_bytes)} -> {human(npy_bytes)}  "
              f"({csv_bytes / npy_bytes:.1f}x)  {time.time() - t0:.0f}s")

    print(f"\npacked {packed}, skipped {skipped}, wrote {human(total_out)} "
          f"in {time.time() - started:.0f}s")
    if not args.delete and packed:
        print("CSVs were kept — rerun with --delete to reclaim the space")


if __name__ == "__main__":
    main()
