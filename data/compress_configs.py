"""
compress_h5.py
==============
Applies sequential cuboid merging to collision maps produced by two pipelines:

  ┌─ efficiency_check.py + combine_npy.py  (--input, HDF5 format)
  │    Input  : collision_configs.h5  { pt_name → uint32[K] }
  │    Output : compressed.h5         { pt_name → uint8[M,10] }
  │    One independent cuboid set per workspace point.
  │    Resumable via checkpoint; suitable for 2+ TB files.
  │
  └─ self_collide_gpu.py  (--input-npy, .npy format)
       Input  : self_collision_map_gpu.npy   uint32[N]  (flat, global)
       Output : self_collision_map_gpu_compressed.npy   uint8[M,10]
       Single cuboid set covering all self-colliding configs.
       No checkpoint needed — runs in one shot.

Each cuboid row (10 bytes):
    [ j1_lo, j1_hi,  j2_lo, j2_hi,  j3_lo, j3_hi,
      j4_lo, j4_hi,  j5_lo, j5_hi ]
All values are joint indices (0-indexed); uint8 is safe since max = 35 < 256.

Decode back to degree arrays (works for both output formats):
    from compress_h5 import storage_to_boxes, boxes_to_degrees
    cuboids   = np.load("...compressed.npy")       # uint8 (M, 10)
    boxes_idx = storage_to_boxes(cuboids)           # int16 (M, 5, 2)  index space
    boxes_deg = boxes_to_degrees(boxes_idx)         # float32 (M, 5, 2) degree space

Usage
-----
  # ── HDF5 mode (efficiency_check.py pipeline) ──────────────────────────────

  # Compare orders on 500 sampled points before full run:
  python compress_h5.py --input collision_configs.h5 --sample 500

  # Full run, default J5→J1:
  python compress_h5.py --input collision_configs.h5 --output compressed.h5

  # Full run, J1→J5:
  python compress_h5.py --input collision_configs.h5 --output compressed_j1j5.h5 \\
                        --order 1,2,3,4,5

  # Resume an interrupted run (checkpoint auto-detected):
  python compress_h5.py --input collision_configs.h5 --output compressed.h5

  # ── .npy mode (self_collide_gpu.py pipeline) ───────────────────────────────

  # Compare both orders on the single self-collision array, then exit:
  python compress_h5.py --input-npy self_collision_map_gpu.npy --sample

  # Full run, default J5→J1:
  python compress_h5.py --input-npy self_collision_map_gpu.npy \\
                        --output self_collision_map_gpu_compressed.npy

  # Full run, J1→J5:
  python compress_h5.py --input-npy self_collision_map_gpu.npy \\
                        --output self_collision_map_gpu_compressed.npy \\
                        --order 1,2,3,4,5
"""

import argparse
import json
import os
import time
import numpy as np
import h5py
from tqdm import tqdm

# ─── Joint space & packing constants ─────────────────────────────────────────
# Must mirror efficiency_check.py / self_collide_gpu.py exactly.
JOINT_DISCRETIZATION = [
    (-180, 180, 10),   # J1: 36 values  ← base rotation
    (-180, 180, 10),   # J2: 36 values
    (-150, 150, 10),   # J3: 30 values
    (-180, 180, 10),   # J4: 36 values
    (-180, 180, 10),   # J5: 36 values  ← wrist
]
SIZES    = np.array([(hi - lo) // st for lo, hi, st in JOINT_DISCRETIZATION])
N_JOINTS = 5
_BITS    = [int(np.ceil(np.log2(n))) if n > 1 else 1 for n in SIZES]   # [6,6,5,6,6]
_SHIFTS  = [sum(_BITS[:i]) for i in range(N_JOINTS)]                    # [0,6,12,17,23]
_MASKS   = [(1 << b) - 1 for b in _BITS]


# ─── Codec ────────────────────────────────────────────────────────────────────
def decode_u32(packed: np.ndarray) -> np.ndarray:
    """(N,) uint32  →  (N, 5) int32  joint indices"""
    p   = packed.view(np.int32)
    idx = np.zeros((len(packed), N_JOINTS), dtype=np.int32)
    for i in range(N_JOINTS):
        idx[:, i] = (p >> _SHIFTS[i]) & _MASKS[i]
    return idx


def storage_to_boxes(data: np.ndarray) -> np.ndarray:
    """(M, 10) uint8  →  (M, 5, 2) int16  [joint, lo/hi]"""
    return data.reshape(-1, N_JOINTS, 2).astype(np.int16)


def boxes_to_storage(boxes: np.ndarray) -> np.ndarray:
    """(M, 5, 2) int16  →  (M, 10) uint8"""
    return boxes.reshape(-1, N_JOINTS * 2).astype(np.uint8)


def boxes_to_degrees(boxes: np.ndarray) -> np.ndarray:
    """
    (M, 5, 2) int16 index space  →  (M, 5, 2) float32 degree space
    Useful for human-readable output or downstream planners.
    """
    out = np.empty_like(boxes, dtype=np.float32)
    for j, (lo, _, st) in enumerate(JOINT_DISCRETIZATION):
        out[:, j, :] = lo + boxes[:, j, :].astype(np.float32) * st
    return out


# ─── Cuboid merging ───────────────────────────────────────────────────────────
def merge_along_dim(boxes: np.ndarray, dim: int) -> np.ndarray:
    """
    Fuse axis-aligned hyper-cuboids along one joint dimension.

    boxes : (N, N_JOINTS, 2) int16  —  boxes[:, j, 0]=lo,  boxes[:, j, 1]=hi

    Merge rule: two boxes fuse iff they are identical in every other dimension
    AND directly adjacent along dim  (hi_a + 1 == lo_b after sorting by lo).
    Adjacency is strict — one missing grid step breaks the chain.
    """
    if len(boxes) == 0:
        return boxes

    other = [i for i in range(boxes.shape[1]) if i != dim]

    # Lexsort: primary = other-dim lo/hi pairs, secondary = current-dim lo
    keys = [boxes[:, dim, 0]]
    for d in reversed(other):
        keys.append(boxes[:, d, 1])
        keys.append(boxes[:, d, 0])
    sb = boxes[np.lexsort(keys)]

    if len(sb) > 1:
        ol           = sb[:, other, 0]
        oh           = sb[:, other, 1]
        group_change = np.any((ol[1:] != ol[:-1]) | (oh[1:] != oh[:-1]), axis=1)
        not_adjacent = sb[1:, dim, 0] != sb[:-1, dim, 1] + 1
        split        = np.concatenate([[True], group_change | not_adjacent])
    else:
        split = np.array([True], dtype=bool)

    starts            = np.where(split)[0]
    ends              = np.concatenate([starts[1:] - 1, [len(sb) - 1]])
    merged            = sb[starts].copy()
    merged[:, dim, 1] = sb[ends, dim, 1]
    return merged


def build_cuboids(idx_5d: np.ndarray, order: list[int]) -> np.ndarray:
    """
    (N, 5) int32  →  (M, 5, 2) int16  hyper-cuboids.
    Sequentially merges along each dimension in `order`.
    Returns empty (0, 5, 2) array if input is empty.
    """
    if len(idx_5d) == 0:
        return np.empty((0, N_JOINTS, 2), dtype=np.int16)
    boxes = np.stack([idx_5d, idx_5d], axis=2).astype(np.int16)
    for dim in order:
        boxes = merge_along_dim(boxes, dim)
    return boxes


def compress_point(packed: np.ndarray, order: list[int]) -> np.ndarray:
    """
    End-to-end compression for one workspace point.
    packed : (K,) uint32  →  (M, 10) uint8
    """
    if len(packed) == 0:
        return np.empty((0, N_JOINTS * 2), dtype=np.uint8)
    return boxes_to_storage(build_cuboids(decode_u32(packed), order))


# ─── Order parsing & display ──────────────────────────────────────────────────
def parse_order(order_str: str) -> list[int]:
    """
    '5,4,3,2,1'  →  [4, 3, 2, 1, 0]   (1-indexed input, 0-indexed output)
    Validates that the result is a complete permutation of 0..N_JOINTS-1.
    """
    parts = [s.strip() for s in order_str.split(",")]
    if len(parts) != N_JOINTS:
        raise ValueError(
            f"--order must list exactly {N_JOINTS} joints, got {len(parts)}: '{order_str}'"
        )
    order = [int(p) - 1 for p in parts]
    if sorted(order) != list(range(N_JOINTS)):
        raise ValueError(
            f"--order must be a permutation of 1–{N_JOINTS}, got '{order_str}'"
        )
    return order


def order_label(order: list[int]) -> str:
    """[4,3,2,1,0]  →  'J5→J4→J3→J2→J1'"""
    return "→".join(f"J{d+1}" for d in order)


# ─── Sample / order-comparison mode ──────────────────────────────────────────
def run_sample(input_h5: str, n_sample: int) -> None:
    """
    Process n_sample random workspace points with every registered candidate
    order, then print a side-by-side comparison to guide the full-run choice.
    """
    candidate_orders = {
        "J5→J4→J3→J2→J1": [4, 3, 2, 1, 0],
        "J1→J2→J3→J4→J5": [0, 1, 2, 3, 4],
    }

    print(f"\nSampling {n_sample} workspace points — order comparison ...")

    with h5py.File(input_h5, "r") as f:
        all_keys = list(f.keys())

    rng  = np.random.default_rng(42)
    keys = list(rng.choice(all_keys, size=min(n_sample, len(all_keys)), replace=False))

    # stats per order: list of (K, M, time_ms) per sampled point
    stats: dict[str, dict] = {
        name: {"K": [], "M": [], "ms": []}
        for name in candidate_orders
    }

    with h5py.File(input_h5, "r") as f:
        for key in tqdm(keys, desc="Sampling", unit="pt"):
            packed = f[key][:].view(np.uint32)
            if len(packed) == 0:
                continue
            for name, order in candidate_orders.items():
                t0      = time.perf_counter()
                cuboids = compress_point(packed, order)
                elapsed = (time.perf_counter() - t0) * 1000
                stats[name]["K"].append(len(packed))
                stats[name]["M"].append(len(cuboids))
                stats[name]["ms"].append(elapsed)

    n_valid = len(stats[next(iter(stats))]["K"])
    print(f"\n{'='*72}")
    print(f"  Order Comparison  ({n_valid} non-empty points sampled)")
    print(f"{'='*72}")
    hdr = (f"  {'Order':<26} {'Avg K':>7} {'Avg M':>7} "
           f"{'Avg ×':>8} {'Med ×':>8} {'p90 ×':>8} {'ms/pt':>7}")
    print(hdr)
    print(f"{'─'*72}")

    for name, s in stats.items():
        K       = np.array(s["K"], dtype=float)
        M       = np.array(s["M"], dtype=float)
        ratios  = np.where(M > 0, K / M, 0.0)
        pcts    = np.percentile(ratios, [50, 90])
        print(
            f"  {name:<26} {K.mean():>7.0f} {M.mean():>7.0f} "
            f"{ratios.mean():>7.1f}x {pcts[0]:>7.1f}x {pcts[1]:>7.1f}x "
            f"{np.mean(s['ms']):>7.2f}"
        )

    print(f"{'='*72}")
    print(f"\n  Ratio percentile breakdown:")
    for name, s in stats.items():
        M      = np.array(s["M"], dtype=float)
        K      = np.array(s["K"], dtype=float)
        ratios = np.where(M > 0, K / M, 0.0)
        p      = np.percentile(ratios, [10, 25, 50, 75, 90])
        print(f"  {name}:")
        print(f"    p10={p[0]:.1f}x  p25={p[1]:.1f}x  p50={p[2]:.1f}x  "
              f"p75={p[3]:.1f}x  p90={p[4]:.1f}x")

    print(f"\n  Points with M==1 (no merge possible):")
    for name, s in stats.items():
        n_trivial = sum(1 for m in s["M"] if m == 1)
        print(f"  {name}: {n_trivial}/{n_valid} "
              f"({100*n_trivial/max(n_valid,1):.1f}%)")


# ─── .npy mode: order comparison ─────────────────────────────────────────────
def run_sample_npy(input_npy: str) -> None:
    """
    Apply both J5→J1 and J1→J5 to the single self-collision uint32 array
    and print a side-by-side comparison.  No --sample N parameter is needed
    here because there is only one array (the full global collision set).
    """
    candidate_orders = {
        "J5→J4→J3→J2→J1": [4, 3, 2, 1, 0],
        "J1→J2→J3→J4→J5": [0, 1, 2, 3, 4],
    }

    print(f"\nLoading '{input_npy}' ...")
    packed = np.load(input_npy).view(np.uint32)
    K      = len(packed)
    print(f"  Loaded {K:,} uint32 values  "
          f"({K * 4 / 1e6:.2f} MB raw)")

    print(f"\nComparing merge orders on full array ({K:,} configs) ...")

    print(f"\n{'='*60}")
    print(f"  Order Comparison  (self_collide_gpu.py output)")
    print(f"{'='*60}")
    print(f"  {'Order':<26} {'K':>10} {'M':>8} {'Ratio':>8} {'Enc (s)':>8}")
    print(f"{'─'*60}")

    for name, order in candidate_orders.items():
        t0      = time.perf_counter()
        cuboids = compress_point(packed, order)
        elapsed = time.perf_counter() - t0
        M       = len(cuboids)
        ratio   = K / M if M > 0 else 0.0
        print(f"  {name:<26} {K:>10,} {M:>8,} {ratio:>7.1f}x {elapsed:>8.2f}")

    print(f"{'='*60}")
    print(f"\nRun again with --output to save the compressed result.")


# ─── .npy mode: full compression run ─────────────────────────────────────────
def run_compress_npy(
    input_npy:  str,
    output_npy: str,
    order:      list[int],
) -> None:
    """
    Compress the flat uint32 array from self_collide_gpu.py into a single
    (M, 10) uint8 .npy file.  No checkpointing is needed — the operation
    is a single pass over one array and completes in seconds.
    """
    label = order_label(order)
    print(f"\n  Merge order : {label}")
    print(f"  Input       : {input_npy}  "
          f"({os.path.getsize(input_npy) / 1e6:.2f} MB)")
    print(f"  Output      : {output_npy}")

    # ── Load ──────────────────────────────────────────────────────────────────
    packed = np.load(input_npy).view(np.uint32)
    K      = len(packed)
    print(f"  Configs (K) : {K:,}")

    if K == 0:
        print("  Input is empty — writing empty output.")
        np.save(output_npy, np.empty((0, N_JOINTS * 2), dtype=np.uint8))
        return

    # ── Compress ──────────────────────────────────────────────────────────────
    print(f"\nCompressing ...")
    t0      = time.perf_counter()
    cuboids = compress_point(packed, order)   # (M, 10) uint8
    elapsed = time.perf_counter() - t0
    M       = len(cuboids)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(output_npy, cuboids)

    # ── Summary ───────────────────────────────────────────────────────────────
    in_bytes  = os.path.getsize(input_npy)
    out_bytes = os.path.getsize(output_npy)
    ratio     = K / M if M > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  Done.   Merge order: {label}")
    print(f"{'='*60}")
    print(f"  Configs  (K) : {K:,}")
    print(f"  Cuboids  (M) : {M:,}")
    print(f"  K/M ratio    : {ratio:.1f}x  (configs per cuboid)")
    print(f"  Input  size  : {in_bytes  / 1e6:.3f} MB")
    print(f"  Output size  : {out_bytes / 1e6:.3f} MB")
    print(f"  Size ratio   : {in_bytes / out_bytes:.1f}x")
    print(f"  Elapsed      : {elapsed:.2f} s")
    print(f"{'='*60}")
    print(f"\nDecode example:")
    print(f"  from compress_h5 import storage_to_boxes, boxes_to_degrees")
    print(f"  cuboids   = np.load('{output_npy}')   # uint8 ({M}, 10)")
    print(f"  boxes_idx = storage_to_boxes(cuboids) # int16 ({M}, 5, 2)")
    print(f"  boxes_deg = boxes_to_degrees(boxes_idx)  # float32 ({M}, 5, 2)")



def load_checkpoint(ckpt_path: str) -> set[str]:
    if not os.path.exists(ckpt_path):
        return set()
    with open(ckpt_path) as f:
        return set(json.load(f).get("done", []))


def save_checkpoint(ckpt_path: str, done: set[str]) -> None:
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"done": sorted(done)}, f)
    os.replace(tmp, ckpt_path)


# ─── Full compression run ─────────────────────────────────────────────────────
def run_compress(
    input_h5:   str,
    output_h5:  str,
    order:      list[int],
    batch_size: int,
    ckpt_path:  str,
) -> None:

    label = order_label(order)
    print(f"\n  Merge order  : {label}")
    print(f"  Input        : {input_h5}  "
          f"({os.path.getsize(input_h5)/1e9:.2f} GB)")
    print(f"  Output       : {output_h5}")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Batch size   : {batch_size} pts per checkpoint flush")

    with h5py.File(input_h5, "r") as f:
        all_keys = list(f.keys())
    print(f"  Total points : {len(all_keys):,}")

    # ── Resume ────────────────────────────────────────────────────────────────
    done    = load_checkpoint(ckpt_path)
    pending = [k for k in all_keys if k not in done]
    print(f"  Already done : {len(done):,}")
    print(f"  Remaining    : {len(pending):,}")

    if not pending:
        print("  All points already compressed — nothing to do.")
        return

    # ── Running accumulators ──────────────────────────────────────────────────
    total_K       = 0    # total input configs across all points
    total_M       = 0    # total output cuboids across all points
    total_skipped = 0    # points with K == 0
    t_start       = time.perf_counter()

    with h5py.File(input_h5,  "r") as h5_in, \
         h5py.File(output_h5, "a") as h5_out:

        # Embed metadata as file-level attributes for traceability
        h5_out.attrs["merge_order"]  = label
        h5_out.attrs["joint_bits"]   = str(_BITS)
        h5_out.attrs["joint_shifts"] = str(_SHIFTS)
        h5_out.attrs["encoding_fmt"] = (
            "uint8 (M, 10): [j1_lo,j1_hi, j2_lo,j2_hi, j3_lo,j3_hi, "
            "j4_lo,j4_hi, j5_lo,j5_hi]  — joint index space"
        )

        pbar = tqdm(
            pending,
            desc=f"Compress ({label})",
            unit="pt",
            dynamic_ncols=True,
        )

        for i, key in enumerate(pbar):
            packed  = h5_in[key][:].view(np.uint32)
            K       = len(packed)

            if K == 0:
                # Preserve the key with an empty placeholder
                if key not in h5_out:
                    h5_out.create_dataset(
                        key,
                        data=np.empty((0, N_JOINTS * 2), dtype=np.uint8)
                    )
                done.add(key)
                total_skipped += 1
                continue

            cuboids = compress_point(packed, order)   # (M, 10) uint8
            M       = len(cuboids)

            total_K += K
            total_M += M

            # Overwrite if resuming and key already exists from a previous run
            if key in h5_out:
                del h5_out[key]
            h5_out.create_dataset(key, data=cuboids, dtype=np.uint8)
            done.add(key)

            # ── Checkpoint flush ──────────────────────────────────────────────
            if (i + 1) % batch_size == 0:
                h5_out.flush()
                save_checkpoint(ckpt_path, done)
                elapsed   = time.perf_counter() - t_start
                avg_ratio = total_K / total_M if total_M > 0 else 0.0
                pts_done  = i + 1
                pts_left  = len(pending) - pts_done
                eta_s     = (elapsed / pts_done) * pts_left if pts_done > 0 else 0
                pbar.set_postfix({
                    "ratio":  f"{avg_ratio:.1f}x",
                    "pts/s":  f"{pts_done / elapsed:.1f}",
                    "ETA":    f"{eta_s/3600:.1f}h",
                })

        # Final flush
        h5_out.flush()
        save_checkpoint(ckpt_path, done)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed     = time.perf_counter() - t_start
    out_bytes   = os.path.getsize(output_h5)
    in_bytes    = os.path.getsize(input_h5)
    avg_ratio   = total_K / total_M if total_M > 0 else 0.0
    n_processed = len(pending) - total_skipped

    print(f"\n{'='*60}")
    print(f"  Done.   Merge order: {label}")
    print(f"{'='*60}")
    print(f"  Points processed  : {n_processed:,}  "
          f"({total_skipped:,} empty skipped)")
    print(f"  Total configs  (K): {total_K:,}")
    print(f"  Total cuboids  (M): {total_M:,}")
    print(f"  Avg K/M ratio     : {avg_ratio:.1f}x  (configs per cuboid)")
    print(f"  Input  HDF5       : {in_bytes/1e9:.3f} GB")
    print(f"  Output HDF5       : {out_bytes/1e9:.3f} GB")
    print(f"  File-size ratio   : {in_bytes/out_bytes:.1f}x")
    print(f"  Elapsed           : {elapsed:.1f} s  "
          f"({elapsed/3600:.2f} hrs)")
    print(f"  Throughput        : {n_processed/elapsed:.1f} pts/s")
    print(f"{'='*60}")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress collision maps via cuboid merging (HDF5 or .npy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input modes (mutually exclusive — provide exactly one):
  --input      HDF5 from efficiency_check.py + combine_npy.py
  --input-npy  .npy  from self_collide_gpu.py

Examples:
  # ── HDF5 mode ──────────────────────────────────────────────────────────────
  python compress_h5.py --input collision_configs.h5 --sample
  python compress_h5.py --input collision_configs.h5 --output compressed.h5
  python compress_h5.py --input collision_configs.h5 --output compressed.h5 --order 1,2,3,4,5

  # ── .npy mode ──────────────────────────────────────────────────────────────
  python compress_h5.py --input-npy self_collision_map_gpu.npy --sample
  python compress_h5.py --input-npy self_collision_map_gpu.npy --output compressed.npy
  python compress_h5.py --input-npy self_collision_map_gpu.npy --output compressed.npy --order 1,2,3,4,5
        """
    )

    # ── Input (mutually exclusive) ─────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Path to collision_configs.h5 from combine_npy.py  (HDF5 mode)"
    )
    input_group.add_argument(
        "--input-npy",
        help="Path to self_collision_map_gpu.npy from self_collide_gpu.py  (.npy mode)"
    )

    # ── Shared options ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output path.  HDF5 mode → .h5 file.  .npy mode → .npy file.  "
            "Required unless --sample is set."
        )
    )
    parser.add_argument(
        "--order", default="5,4,3,2,1",
        help=(
            "Merge order: comma-separated 1-indexed joint numbers. "
            "Default '5,4,3,2,1' (J5→J1, wrist-first). "
            "Use '1,2,3,4,5' for J1→J5 (base-first). "
            "Any permutation of 1–5 is valid."
        )
    )
    parser.add_argument(
        "--sample", action="store_true",
        help=(
            "Order-comparison mode — report stats without writing output. "
            "HDF5 mode: samples --sample-size random points. "
            ".npy mode: runs on the full array (no size needed)."
        )
    )
    parser.add_argument(
        "--sample-size", type=int, default=500, metavar="N",
        help="(HDF5 mode only) Number of random workspace points to sample. Default: 500."
    )

    # ── HDF5-only options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="(HDF5 mode only) Flush checkpoint every N points. Default: 1000."
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="(HDF5 mode only) Checkpoint JSON path. Default: <o>.ckpt.json"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Collision Map Cuboid Compressor")
    print("=" * 60)
    print(f"  Joint sizes   : {list(SIZES)}")
    print(f"  uint32 packing: {_BITS} bits, shifts={_SHIFTS}")

    # ── .npy mode ─────────────────────────────────────────────────────────────
    if args.input_npy is not None:
        if args.sample:
            run_sample_npy(args.input_npy)
            return
        if args.output is None:
            parser.error("--output is required in .npy mode when not using --sample")
        order = parse_order(args.order)
        run_compress_npy(args.input_npy, args.output, order)
        return

    # ── HDF5 mode ─────────────────────────────────────────────────────────────
    if args.sample:
        run_sample(args.input, args.sample_size)
        return
    if args.output is None:
        parser.error("--output is required in HDF5 mode when not using --sample")
    order = parse_order(args.order)
    ckpt  = args.checkpoint or (args.output + ".ckpt.json")
    run_compress(args.input, args.output, order, args.batch_size, ckpt)


if __name__ == "__main__":
    main()
