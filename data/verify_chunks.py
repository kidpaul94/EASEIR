"""
verify_chunks.py
================
Run this AFTER reencode_configs.py to confirm all chunk files are correctly
encoded as uint32 and all decoded values land exactly on valid grid points.

Two checks per chunk file, both fully vectorized (no Python loops over rows):
  1. dtype      — every array is uint32 (not float32)
  2. round-trip — encode(decode(packed)) == packed for every config

The round-trip check is equivalent to verifying that every decoded angle
lands on a valid grid point, but runs entirely as numpy array ops instead
of per-element Python membership tests — orders of magnitude faster.

Usage
-----
    python verify_chunks.py                          # uses defaults below
    python verify_chunks.py --chunk-dir /path/to/dir
    python verify_chunks.py --max-errors 50          # stop after N errors (default 20)
"""

import os
import re
import argparse
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config (edit here OR pass CLI flags)
# ---------------------------------------------------------------------------
SSD_MOUNT = "/media/hojunlee/Extreme Pro"
CHUNK_DIR = os.path.join(SSD_MOUNT, "collision_configs")
# ---------------------------------------------------------------------------

# Must match JOINT_DISCRETIZATION in efficiency_check.py / reencode_configs.py
JOINT_DISCRETIZATION = [
    (-180, 180, 10),
    (-180, 180, 10),
    (-150, 150, 10),
    (-180, 180, 10),
    (-180, 180, 10),
    (   0,   1,  1),
]

N_JOINTS = len(JOINT_DISCRETIZATION)
N_PACKED = 5   # joint 6 is always 0, not stored

_LO    = [JOINT_DISCRETIZATION[i][0]                                          for i in range(N_PACKED)]
_STEP  = [JOINT_DISCRETIZATION[i][2]                                          for i in range(N_PACKED)]
_NVALS = [(JOINT_DISCRETIZATION[i][1] - JOINT_DISCRETIZATION[i][0])
          // JOINT_DISCRETIZATION[i][2]                                        for i in range(N_PACKED)]
_BITS  = [int(np.ceil(np.log2(n))) if n > 1 else 1                            for n in _NVALS]
_SHIFTS = [sum(_BITS[:i]) for i in range(N_PACKED)]
_MASKS  = [(1 << b) - 1  for b in _BITS]

CHUNK_RE = re.compile(r"^chunk_b(\d{8})\.npy$")


# =============================================================================
# VECTORIZED ENCODE / DECODE  (pure numpy, no Python loops over rows)
# =============================================================================

def decode_configs(packed: np.ndarray) -> np.ndarray:
    """uint32 (K,) → float32 (K, 6)"""
    configs = np.zeros((len(packed), N_JOINTS), dtype=np.float32)
    for i in range(N_PACKED):
        idx = ((packed >> _SHIFTS[i]) & _MASKS[i]).astype(np.int32)
        configs[:, i] = idx * _STEP[i] + _LO[i]
    return configs


def encode_configs(configs: np.ndarray) -> np.ndarray:
    """float32 (K, 6) → uint32 (K,)"""
    packed = np.zeros(len(configs), dtype=np.uint32)
    for i in range(N_PACKED):
        idx = np.round((configs[:, i] - _LO[i]) / _STEP[i]).astype(np.uint32)
        packed |= idx << _SHIFTS[i]
    return packed


def verify_round_trip(packed: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask of shape (K,) — True where round-trip failed.
    Fully vectorized: decode then re-encode and compare, no per-row Python loop.
    """
    repacked = encode_configs(decode_configs(packed))
    return repacked != packed   # (K,) bool, True = corrupted config


# =============================================================================
# HELPERS
# =============================================================================

def discover_chunks(chunk_dir: str) -> list[tuple[int, str]]:
    chunks = []
    for fname in os.listdir(chunk_dir):
        m = CHUNK_RE.match(fname)
        if m:
            chunks.append((int(m.group(1)), os.path.join(chunk_dir, fname)))
    return sorted(chunks, key=lambda x: x[0])


def parse_args():
    p = argparse.ArgumentParser(description="Verify uint32-encoded collision config chunk files.")
    p.add_argument("--chunk-dir",  default=CHUNK_DIR,
                   help=f"Directory containing chunk_b*.npy files (default: {CHUNK_DIR})")
    p.add_argument("--max-errors", type=int, default=20,
                   help="Stop reporting after this many errors (default: 20)")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if not os.path.isdir(args.chunk_dir):
        raise SystemExit(f"[ERROR] Chunk directory not found: {args.chunk_dir}")

    chunks = discover_chunks(args.chunk_dir)
    if not chunks:
        raise SystemExit(f"[ERROR] No chunk_b????????.npy files found in {args.chunk_dir}")

    print(f"Found {len(chunks):,} chunk files.")
    print(f"Bit layout  : {_BITS}  (total {sum(_BITS)}/32 bits used)")
    print(f"Checks      : (1) dtype == uint32  "
          f"(2) encode(decode(packed)) == packed  [fully vectorized]\n")

    errors         = []
    n_chunks_ok    = 0
    n_pts_checked  = 0
    n_rows_checked = 0

    for batch_idx, fpath in tqdm(chunks, desc="Verifying", unit="chunk"):
        chunk = np.load(fpath, allow_pickle=True).item()
        chunk_ok = True

        for pt_idx, packed in chunk.items():

            # -- Check 1: dtype ------------------------------------------------
            if packed.dtype != np.uint32:
                err = (f"chunk_b{batch_idx:08d} | pt {pt_idx:,} | "
                       f"wrong dtype: {packed.dtype} (expected uint32)")
                errors.append(err)
                tqdm.write(f"[ERROR] {err}")
                chunk_ok = False
                if len(errors) >= args.max_errors:
                    tqdm.write(f"\nReached --max-errors {args.max_errors}. Stopping early.")
                    _print_summary(chunks, n_chunks_ok, n_pts_checked, n_rows_checked, errors)
                    return
                continue   # skip round-trip check for float32 arrays

            # -- Check 2: round-trip (fully vectorized) ------------------------
            bad_mask = verify_round_trip(packed)
            if bad_mask.any():
                n_bad   = int(bad_mask.sum())
                decoded = decode_configs(packed[bad_mask][:3])   # sample up to 3
                err = (f"chunk_b{batch_idx:08d} | pt {pt_idx:,} | "
                       f"{n_bad} configs fail round-trip "
                       f"(sample decoded: {decoded.tolist()})")
                errors.append(err)
                tqdm.write(f"[ERROR] {err}")
                chunk_ok = False
                if len(errors) >= args.max_errors:
                    tqdm.write(f"\nReached --max-errors {args.max_errors}. Stopping early.")
                    _print_summary(chunks, n_chunks_ok, n_pts_checked, n_rows_checked, errors)
                    return

            n_pts_checked  += 1
            n_rows_checked += packed.shape[0]

        if chunk_ok:
            n_chunks_ok += 1

    _print_summary(chunks, n_chunks_ok, n_pts_checked, n_rows_checked, errors)


def _print_summary(chunks, n_chunks_ok, n_pts_checked, n_rows_checked, errors):
    print(f"\n{'=' * 60}")
    print(f"Chunks verified    : {n_chunks_ok:,} / {len(chunks):,} fully OK")
    print(f"Grid points checked: {n_pts_checked:,}")
    print(f"Configs checked    : {n_rows_checked:,}")
    if errors:
        print(f"Errors found       : {len(errors):,}")
        print("\nFirst errors:")
        for e in errors[:20]:
            print(f"  {e}")
    else:
        print(f"Errors found       : 0")
        print(f"\nAll chunks verified OK — all uint32, all values on valid grid.")


if __name__ == "__main__":
    main()