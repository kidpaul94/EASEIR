"""
combine_npy.py
==============
Run this AFTER efficiency_check.py (chunks are already uint32-encoded).

Each chunk file is a dict:  { pt_idx (int) -> uint32 array, shape (K,) }

Writes a single HDF5 file where each dataset is named after the point's
3D coordinates and contains all collision configs for that point:

    {OUTPUT_H5}
        /{pt_name}    e.g. /n0p7790_n0p7790_n0p6440
                      uint32, shape (K_total,)

Disk usage stays flat throughout: each buffer flush writes ~N×280MB to HDF5
and deletes ~N×280MB of chunk files, so net disk delta ≈ 0.

    Chunk files (2.2TB, shrinking)
    HDF5 file   (2.2TB, growing)
    ─────────────────────────────
    Peak SSD usage ≈ 2.2TB + one buffer window (~4.2GB headroom)

Memory usage:
    COMBINE_BUFFER_CHUNKS × 280MB ≈ 4.2GB peak RAM  (15 chunks default)

Decode back to float32 angles using decode_configs() from reencode_configs.py:

    import h5py
    from reencode_configs import decode_configs
    with h5py.File("collision_configs.h5", "r") as f:
        packed  = f["n0p7790_n0p7790_n0p6440"][:]   # uint32 (K,)
    configs = decode_configs(packed)                 # float32 (K, 6)

Resumable: tracks progress in combine_checkpoint.json. Safe to rerun after
interruption — already-processed chunks are skipped, existing HDF5 datasets
are extended rather than overwritten.

Note: chunk files are written by two GPU workers and follow the pattern
      chunk_g{rank}_b{batch_idx:08d}.npy  (e.g. chunk_g0_b00000000.npy).
      Both workers' files are discovered and merged into the same HDF5.
      New chunk files are serialised with pickle; the loader also handles
      legacy files written with np.save for backward compatibility.

Usage
-----
    python combine_npy.py                              # uses defaults below
    python combine_npy.py --output-dir /path/to/dir
    python combine_npy.py --keep-chunks                # skip deletion of chunks
    python combine_npy.py --dry-run                    # report only, no writes
    python combine_npy.py --buffer-chunks 15           # tune for available RAM
                                                       # rule: N × 280MB < RAM × 0.7
"""

import os
import re
import json
import pickle
import argparse
import numpy as np
import h5py
from collections import defaultdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config (edit here OR pass CLI flags)
# ---------------------------------------------------------------------------
SSD_MOUNT  = "/media/hojunlee/T7 Shield"
CHUNK_DIR  = os.path.join(SSD_MOUNT, "collision_configs")
OUTPUT_H5  = os.path.join("./", "collision_configs.h5")

COMBINE_BUFFER_CHUNKS = 1
# ---------------------------------------------------------------------------

# Matches both GPU workers: chunk_g0_b00000000.npy, chunk_g1_b00000000.npy
CHUNK_RE     = re.compile(r"^chunk_g(\d+)_b(\d{8})\.npy$")
COMBINE_CKPT = os.path.join(CHUNK_DIR, "combine_checkpoint.json")


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge uint32 chunk files into a single HDF5 file.")
    p.add_argument("--chunk-dir",     default=CHUNK_DIR,
                   help=f"Directory containing chunk_g*_b*.npy files "
                        f"(default: {CHUNK_DIR})")
    p.add_argument("--output-h5",     default=OUTPUT_H5,
                   help=f"Path for output HDF5 file (default: {OUTPUT_H5})")
    p.add_argument("--keep-chunks",   action="store_true",
                   help="Do NOT delete chunk files after processing.")
    p.add_argument("--dry-run",       action="store_true",
                   help="Report what would happen without writing files.")
    p.add_argument("--buffer-chunks", type=int, default=COMBINE_BUFFER_CHUNKS,
                   help=f"Chunks to buffer before flushing to HDF5 "
                        f"(default: {COMBINE_BUFFER_CHUNKS}). "
                        f"Rule: N × 280MB < available_RAM_MB × 0.7")
    return p.parse_args()


# =============================================================================
# CHUNK LOADER  (handles both pickle and legacy np.save formats)
# =============================================================================

def load_chunk(fpath: str) -> dict:
    """
    Load a chunk file written by efficiency_check.py.

    np.save on a plain Python dict produces a 0-d object array whose payload
    is a pickle stream written after the numpy header. The header-length field
    is 2 bytes (uint16) for NPY version 1.x and 4 bytes (uint32) for version
    2.x (the default in NumPy ≥ 2.0 for large headers). We detect the version
    and adjust accordingly.

    Byte layout of a numpy .npy file:
        bytes 0-5              : magic  b'\x93NUMPY'
        bytes 6                : major version
        bytes 7                : minor version
        bytes 8-9  (v1.x)     : header_len as uint16
        bytes 8-11 (v2.x+)    : header_len as uint32
        bytes (10 or 12)-(10 or 12 + header_len - 1) : ASCII header dict
        remaining bytes        : raw pickle payload  (0-d object array)
    """
    # Strategy 1: direct pickle (future files written with pickle.dump)
    try:
        with open(fpath, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # Strategy 2: strip the numpy header and unpickle the payload directly.
    # Handles both NPY v1.x (uint16 header_len) and v2.x+ (uint32 header_len).
    # The result may be a 0-d numpy object array — unwrap it with .item().
    try:
        with open(fpath, "rb") as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                raise ValueError("Not a numpy file")
            major = int.from_bytes(f.read(1), "little")        # major version
            f.read(1)                                           # minor version (unused)
            if major >= 2:
                header_len = int.from_bytes(f.read(4), "little")   # uint32 for v2+
            else:
                header_len = int.from_bytes(f.read(2), "little")   # uint16 for v1.x
            f.read(header_len)                                  # skip header dict
            result = pickle.load(f)                             # unpickle payload
        # np.save wraps plain objects in a 0-d ndarray — unwrap if needed
        if isinstance(result, np.ndarray) and result.ndim == 0:
            result = result.item()
        if isinstance(result, dict):
            return result
        raise ValueError(f"Expected dict, got {type(result)}")
    except Exception:
        pass

    # Strategy 3: let numpy's own loader handle it (most compatible fallback).
    # np.load returns a 0-d object array for dicts saved with np.save; unwrap
    # it with .item().
    try:
        result = np.load(fpath, allow_pickle=True)
        if isinstance(result, np.ndarray) and result.ndim == 0:
            result = result.item()
        if isinstance(result, dict):
            return result
        raise ValueError(f"Expected dict, got {type(result)}")
    except Exception as e:
        raise RuntimeError(
            f"Could not load chunk file '{fpath}'. "
            f"The file may be corrupted. Original error: {e}"
        )


# =============================================================================
# CHECKPOINT  (tracks individual (rank, batch_idx) pairs)
# =============================================================================

def load_checkpoint() -> set[tuple[int, int]]:
    """Return a set of (rank, batch_idx) pairs already combined."""
    if not os.path.exists(COMBINE_CKPT):
        return set()
    with open(COMBINE_CKPT) as f:
        data = json.load(f)
    return {tuple(pair) for pair in data.get("combined_chunks", [])}


def save_checkpoint(combined: set[tuple[int, int]]) -> None:
    tmp = COMBINE_CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"combined_chunks": sorted(combined)}, f, indent=2)
    os.replace(tmp, COMBINE_CKPT)


# =============================================================================
# PT_NAMES  (recomputed — no dependency on efficiency_check.py output)
# =============================================================================

def build_pt_names() -> list[str]:
    import torch
    step    = 0.01
    center  = torch.tensor([0.0, 0.0, 0.135])
    r_outer = 1.558 / 2
    r_inner = 0.411 / 2

    pts_all = torch.cartesian_prod(
        torch.arange(-r_outer, r_outer + step, step),
        torch.arange(-r_outer, r_outer + step, step),
        torch.arange(center[2] - r_outer, center[2] + r_outer + step, step))
    dist_sq = ((pts_all - center) ** 2).sum(-1)
    pts_cpu = pts_all[
        (dist_sq <= r_outer ** 2) & (dist_sq >= r_inner ** 2)
    ].numpy()

    def coord_to_str(v: float) -> str:
        sign = "n" if v < 0 else ""
        return sign + f"{abs(v):.4f}".replace(".", "p")

    return [
        f"{coord_to_str(x)}_{coord_to_str(y)}_{coord_to_str(z)}"
        for x, y, z in pts_cpu
    ]


# =============================================================================
# FLUSH BUFFER → HDF5
# =============================================================================

def flush_buffer(
    buffer:           defaultdict,
    h5_file:          h5py.File,
    pt_names:         list,
    chunks_in_buffer: list[tuple[int, int, str]],   # (rank, batch_idx, fpath)
    keep_chunks:      bool,
    combined:         set[tuple[int, int]],
) -> tuple[int, int]:
    """
    Write all buffered pt_idx arrays into the HDF5 file.
    If a dataset already exists (resume case), extend it.
    Then checkpoint and optionally delete the chunk files.

    Returns (n_pts_flushed, n_rows_flushed).
    """
    n_pts = n_rows = 0

    last_rank, last_batch_idx, _ = chunks_in_buffer[-1]

    for pt_idx, arrays in tqdm(buffer.items(),
                               desc=f"Flushing (GPU {last_rank} batch {last_batch_idx:,})",
                               unit="pt", dynamic_ncols=True, leave=False):
        name     = pt_names[pt_idx]
        new_data = np.concatenate(arrays, axis=0)   # uint32 (K,)
        del arrays

        if name in h5_file:
            # Resume case: extend existing dataset
            ds      = h5_file[name]
            old_len = ds.shape[0]
            new_len = old_len + new_data.shape[0]
            ds.resize(new_len, axis=0)
            ds[old_len:new_len] = new_data
        else:
            # First write: create resizable dataset
            h5_file.create_dataset(
                name,
                data=new_data,
                maxshape=(None,),
                chunks=(min(4096, len(new_data)),),
            )

        n_pts  += 1
        n_rows += new_data.shape[0]
        del new_data

    h5_file.flush()

    # Mark every chunk in this window as combined, then checkpoint
    for rank, batch_idx, _ in chunks_in_buffer:
        combined.add((rank, batch_idx))
    save_checkpoint(combined)

    if not keep_chunks:
        for _, _, fpath in chunks_in_buffer:
            os.remove(fpath)
        tqdm.write(
            f"[deleted] {len(chunks_in_buffer):,} chunk files "
            f"(up to GPU {last_rank} batch {last_batch_idx:,})",
        )

    return n_pts, n_rows


# =============================================================================
# DISCOVERY
# =============================================================================

def discover_chunks(chunk_dir: str) -> list[tuple[int, int, str]]:
    """Return list of (rank, batch_idx, fpath) sorted by (batch_idx, rank)."""
    chunks = []
    for fname in os.listdir(chunk_dir):
        m = CHUNK_RE.match(fname)
        if m:
            rank      = int(m.group(1))
            batch_idx = int(m.group(2))
            chunks.append((rank, batch_idx, os.path.join(chunk_dir, fname)))
    return sorted(chunks, key=lambda x: (x[1], x[0]))


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if not os.path.isdir(args.chunk_dir):
        raise SystemExit(f"[ERROR] Chunk directory not found: {args.chunk_dir}")

    # -- Build pt_names --------------------------------------------------------
    print("Building point name lookup...")
    pt_names = build_pt_names()
    print(f"  {len(pt_names):,} points")
    print(f"  Example: pt_idx=0 → '{pt_names[0]}'")

    # -- Discover chunks -------------------------------------------------------
    chunks = discover_chunks(args.chunk_dir)
    if not chunks:
        print("No chunk files (chunk_g*_b????????.npy) found. Nothing to do.")
        return

    # -- Resume ----------------------------------------------------------------
    combined  = load_checkpoint()
    pending   = [(r, bi, fp) for r, bi, fp in chunks if (r, bi) not in combined]
    n_done    = len(chunks) - len(pending)

    print(f"\nChunk files found  : {len(chunks):,}  (GPU 0: "
          f"{sum(1 for r, _, _ in chunks if r == 0):,}, "
          f"GPU 1: {sum(1 for r, _, _ in chunks if r == 1):,})")
    print(f"Already combined   : {n_done:,}")
    print(f"Remaining          : {len(pending):,}")
    print(f"Buffer size        : {args.buffer_chunks} chunks  "
          f"(~{args.buffer_chunks * 0.28:.1f} GB estimated peak RAM)")
    print(f"Output HDF5        : {args.output_h5}")

    if args.dry_run:
        print("[DRY RUN] No files will be written or deleted.")

    if not pending:
        print("All chunks already combined. Nothing to do.")
        return

    # -- Open HDF5 (append mode for resume safety) -----------------------------
    total_pts_touched: set[int] = set()
    total_rows = 0

    with h5py.File(args.output_h5, "a") as h5_file:

        buffer: defaultdict[int, list[np.ndarray]] = defaultdict(list)
        buffer_window: list[tuple[int, int, str]] = []

        pbar = tqdm(pending, desc="Loading chunks",
                    unit="chunk", dynamic_ncols=True)

        for rank, batch_idx, fpath in pbar:
            chunk = load_chunk(fpath)

            for pt_idx, packed in chunk.items():
                buffer[pt_idx].append(packed)
                total_pts_touched.add(pt_idx)
                total_rows += packed.shape[0]

            buffer_window.append((rank, batch_idx, fpath))

            if len(buffer_window) >= args.buffer_chunks:
                if not args.dry_run:
                    n_pts, _ = flush_buffer(
                        buffer, h5_file, pt_names,
                        buffer_window, args.keep_chunks, combined)
                    pbar.set_postfix({"flushed pts": f"{n_pts:,}"})
                buffer.clear()
                buffer_window.clear()

        # Flush remaining partial window
        if buffer_window and not args.dry_run:
            flush_buffer(
                buffer, h5_file, pt_names,
                buffer_window, args.keep_chunks, combined)

    # -- Summary ---------------------------------------------------------------
    h5_size = os.path.getsize(args.output_h5) / 1e9 if \
              os.path.exists(args.output_h5) else 0

    print(f"\nDone.")
    print(f"  Chunks processed    : {len(pending):,}")
    print(f"  Grid points touched : {len(total_pts_touched):,}")
    print(f"  Total configs       : {total_rows:,}")
    print(f"  HDF5 size on disk   : {h5_size:.2f} GB")
    if args.dry_run:
        print("  [DRY RUN] No actual changes made.")
    elif not args.keep_chunks:
        print(f"  Chunk files deleted : {len(pending):,}")
    print(f"\nTo load a point's configs:")
    print(f"  import h5py")
    print(f"  from reencode_configs import decode_configs")
    print(f"  with h5py.File('{args.output_h5}', 'r') as f:")
    print(f"      packed  = f['<pt_name>'][:]   # uint32 (K,)")
    print(f"  configs = decode_configs(packed)  # float32 (K, 6)")


if __name__ == "__main__":
    main()