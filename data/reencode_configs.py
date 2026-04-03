"""
reencode_configs.py
===================
Run this AFTER efficiency_check.py, BEFORE combine_npy.py.

Reads each chunk file produced by efficiency_check.py:

    {CHUNK_DIR}/chunk_b{batch_idx:08d}.npy     dict { pt_idx -> float32 (K, 6) }

Re-encodes each float32 config array to a packed uint32 array in-place,
replacing the chunk files with their uint32 equivalents:

    {CHUNK_DIR}/chunk_b{batch_idx:08d}.npy     dict { pt_idx -> uint32  (K,)  }

combine_npy.py then combines the uint32 chunks into final per-point files:

    {CHUNK_DIR}/pt{pt_idx:08d}.npy             uint32, shape (K_total,)

This means the full float32 per-point files are NEVER written to disk,
saving 6x storage compared to going through float32 → combine → reencode.

Bit layout (all 5 meaningful joints packed into one uint32):
    bits  0– 5 : joint 1 index  (0–35, maps to –180..170 deg, step 10)
    bits  6–11 : joint 2 index  (0–35, maps to –180..170 deg, step 10)
    bits 12–16 : joint 3 index  (0–29, maps to –150..140 deg, step 10)
    bits 17–22 : joint 4 index  (0–35, maps to –180..170 deg, step 10)
    bits 23–28 : joint 5 index  (0–35, maps to –180..170 deg, step 10)
    bits 29–31 : unused (always 0)
    joint 6 is always 0 and is dropped; decode_configs() restores it as 0.0

Usage
-----
    python reencode_configs.py                        # uses defaults below
    python reencode_configs.py --chunk-dir /path/to/collision_configs
    python reencode_configs.py --verify               # decode and check round-trip
    python reencode_configs.py --dry-run              # report only, no writes
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

# Must match JOINT_DISCRETIZATION in efficiency_check.py
JOINT_DISCRETIZATION = [
    (-180, 180, 10),
    (-180, 180, 10),
    (-150, 150, 10),
    (-180, 180, 10),
    (-180, 180, 10),
    (   0,   1,  1),   # always 0 — dropped from packed representation
]

N_JOINTS = len(JOINT_DISCRETIZATION)
N_PACKED = 5           # joint 6 is constant → not stored

_LO    = [JOINT_DISCRETIZATION[i][0]                                          for i in range(N_PACKED)]
_STEP  = [JOINT_DISCRETIZATION[i][2]                                          for i in range(N_PACKED)]
_NVALS = [(JOINT_DISCRETIZATION[i][1] - JOINT_DISCRETIZATION[i][0])
          // JOINT_DISCRETIZATION[i][2]                                        for i in range(N_PACKED)]
_BITS  = [int(np.ceil(np.log2(n))) if n > 1 else 1                            for n in _NVALS]
_SHIFTS = [sum(_BITS[:i]) for i in range(N_PACKED)]
_MASKS  = [(1 << b) - 1  for b in _BITS]

assert sum(_BITS) <= 32, (
    f"Packed representation needs {sum(_BITS)} bits — exceeds uint32. "
    "Increase dtype to uint64 or reduce discretization."
)

print(f"Bit layout  : {_BITS}  (total {sum(_BITS)}/32 bits used)")
print(f"Shifts      : {_SHIFTS}")
print(f"N values    : {_NVALS}")
print(f"Compression : {N_JOINTS * 4} bytes/config (float32) → 4 bytes/config (uint32)  "
      f"({N_JOINTS}× reduction)\n")


# =============================================================================
# ENCODE / DECODE
# =============================================================================

def encode_configs(configs: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    configs : float32 array, shape (K, 6)
        Actual joint angles in degrees.

    Returns
    -------
    packed : uint32 array, shape (K,)
    """
    packed = np.zeros(len(configs), dtype=np.uint32)
    for i in range(N_PACKED):
        idx = np.round((configs[:, i] - _LO[i]) / _STEP[i]).astype(np.uint32)
        packed |= idx << _SHIFTS[i]
    return packed


def decode_configs(packed: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    packed : uint32 array, shape (K,)

    Returns
    -------
    configs : float32 array, shape (K, 6)
        Reconstructed joint angles in degrees.
    """
    configs = np.zeros((len(packed), N_JOINTS), dtype=np.float32)
    for i in range(N_PACKED):
        idx = ((packed >> _SHIFTS[i]) & _MASKS[i]).astype(np.int32)  # cast before negative arithmetic
        configs[:, i] = idx * _STEP[i] + _LO[i]
    # configs[:, 5] stays 0.0 — joint 6 is always 0
    return configs


def verify_round_trip(original: np.ndarray, packed: np.ndarray) -> bool:
    reconstructed = decode_configs(packed)
    max_err = np.abs(original[:, :N_PACKED] - reconstructed[:, :N_PACKED]).max()
    return max_err < 0.5   # exact grid values should round-trip with < 1e-3 error


# =============================================================================
# MAIN
# =============================================================================

CHUNK_RE = re.compile(r"^chunk_b(\d{8})\.npy$")

def discover_chunks(chunk_dir: str) -> list[tuple[int, str]]:
    chunks = []
    for fname in os.listdir(chunk_dir):
        m = CHUNK_RE.match(fname)
        if m:
            chunks.append((int(m.group(1)), os.path.join(chunk_dir, fname)))
    return sorted(chunks, key=lambda x: x[0])


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-encode chunk files from float32 to packed uint32 in-place.")
    p.add_argument("--chunk-dir", default=CHUNK_DIR,
                   help=f"Directory containing chunk_b*.npy files (default: {CHUNK_DIR})")
    p.add_argument("--verify",    action="store_true",
                   help="Decode each encoded chunk and verify round-trip accuracy.")
    p.add_argument("--dry-run",   action="store_true",
                   help="Report sizes without writing files.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.chunk_dir):
        raise SystemExit(f"[ERROR] Chunk directory not found: {args.chunk_dir}")

    chunks = discover_chunks(args.chunk_dir)
    if not chunks:
        raise SystemExit(
            f"[ERROR] No chunk_b????????.npy files found in {args.chunk_dir}\n"
            "        Make sure you run this before combine_npy.py."
        )

    print(f"Found {len(chunks):,} chunk files in {args.chunk_dir}")
    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    total_in_bytes  = 0
    total_out_bytes = 0
    verify_errors   = []
    skipped         = 0

    for batch_idx, fpath in tqdm(chunks, desc="Re-encoding chunks", unit="chunk"):
        chunk_in: dict = np.load(fpath, allow_pickle=True).item()

        # --- Safe resume: skip files already converted to uint32 --------------
        first_val = next(iter(chunk_in.values()))
        if first_val.dtype == np.uint32:
            total_out_bytes += sum(a.nbytes for a in chunk_in.values())
            skipped += 1
            continue

        # --- Encode float32 → uint32 ------------------------------------------
        chunk_u32: dict[int, np.ndarray] = {}
        for pt_idx, configs in chunk_in.items():
            packed = encode_configs(configs)

            if args.verify:
                if not verify_round_trip(configs, packed):
                    verify_errors.append((batch_idx, pt_idx))
                    tqdm.write(
                        f"[WARN] Round-trip error: chunk_b{batch_idx:08d}, pt {pt_idx}"
                    )

            total_in_bytes  += configs.nbytes
            total_out_bytes += packed.nbytes
            chunk_u32[pt_idx] = packed

        if not args.dry_run:
            # Atomic in-place replacement
            tmp = fpath + ".tmp.npy"
            np.save(tmp, chunk_u32)
            os.replace(tmp, fpath)

    # -- Summary ---------------------------------------------------------------
    ratio = total_in_bytes / total_out_bytes if total_out_bytes else 0
    print(f"\nDone.")
    print(f"  Chunk files found     : {len(chunks):,}")
    print(f"  Already encoded       : {skipped:,}  (skipped)")
    print(f"  Newly encoded         : {len(chunks) - skipped:,}")
    print(f"  Input  size (float32) : {total_in_bytes  / 1e9:.3f} GB")
    print(f"  Output size (uint32)  : {total_out_bytes / 1e9:.3f} GB")
    if total_in_bytes > 0:
        print(f"  Compression ratio     : {ratio:.2f}×")
    if args.verify:
        if verify_errors:
            print(f"  Round-trip errors     : {len(verify_errors)} — check WARN messages above")
        else:
            print(f"  Round-trip verify     : all OK")
    if args.dry_run:
        print("  [DRY RUN] No files written.")
    if not args.dry_run:
        print("\nNext step: run combine_npy.py to merge chunks into per-point files.")


if __name__ == "__main__":
    main()


# =============================================================================
# COPY THIS INTO ANY DOWNSTREAM SCRIPT THAT LOADS THE FINAL PER-POINT FILES
# =============================================================================
#
# from reencode_configs import decode_configs
#
# packed  = np.load("pt00000000.npy")          # uint32, shape (K,)
# configs = decode_configs(packed)             # float32, shape (K, 6)
#