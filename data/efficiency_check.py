import torch
import time
import json
import numpy as np
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import os
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm

# =============================================================================
# GLOBAL CONSTANTS  (CPU-only — safe to define before forking)
# =============================================================================

root_path  = "/home/hojunlee/Work/EASEIR++/test"
robot_name = "m0609_wo_eef"

# --- Workspace pts (built on CPU, moved to GPU inside each worker) -----------
step    = 0.01
CENTER  = torch.tensor([0.0, 0.0, 0.135])   # CPU
r_outer = 1.558 / 2
r_inner = 0.411 / 2

_pts_all = torch.cartesian_prod(
    torch.arange(-r_outer, r_outer + step, step),
    torch.arange(-r_outer, r_outer + step, step),
    torch.arange(CENTER[2] - r_outer, CENTER[2] + r_outer + step, step))
_dist_sq = ((_pts_all - CENTER) ** 2).sum(-1)
PTS_CPU  = _pts_all[(_dist_sq <= r_outer ** 2) & (_dist_sq >= r_inner ** 2)]   # float32, CPU
N_pts    = len(PTS_CPU)

def coord_to_str(v: float) -> str:
    sign = "n" if v < 0 else ""
    return sign + f"{abs(v):.4f}".replace(".", "p")

pts_np   = PTS_CPU.numpy()
pt_names = [
    f"{coord_to_str(x)}_{coord_to_str(y)}_{coord_to_str(z)}"
    for x, y, z in pts_np
]

# -- Discretized joint limits --------------------------------------------------
JOINT_DISCRETIZATION = [
    (-180, 180, 10),
    (-180, 180, 10),
    (-150, 150, 10),
    (-180, 180, 10),
    (-180, 180, 10),
    (   0,   1, 1),
]

grids           = [torch.arange(lo, hi, st, dtype=torch.float32) for lo, hi, st in JOINT_DISCRETIZATION]
N_joints        = len(grids)
N_per_joint     = torch.tensor([len(g) for g in grids], dtype=torch.long)
N_total         = N_per_joint.prod().item()
strides         = torch.ones(N_joints, dtype=torch.long)
for i in range(N_joints - 2, -1, -1):
    strides[i]  = strides[i + 1] * N_per_joint[i + 1]
strides_int     = strides.tolist()
N_per_joint_int = N_per_joint.tolist()

# -- GPU encoding constants (CPU tensors — cloned to device inside worker) ----
N_PACKED    = 5
_ENC_LO_CPU   = torch.tensor([jd[0] for jd in JOINT_DISCRETIZATION[:N_PACKED]], dtype=torch.float32)
_ENC_STEP_CPU = torch.tensor([jd[2] for jd in JOINT_DISCRETIZATION[:N_PACKED]], dtype=torch.float32)
_NVALS        = [(jd[1] - jd[0]) // jd[2] for jd in JOINT_DISCRETIZATION[:N_PACKED]]
_BITS         = [int(np.ceil(np.log2(n))) if n > 1 else 1 for n in _NVALS]
_ENC_SHIFTS   = [sum(_BITS[:i]) for i in range(N_PACKED)]
assert sum(_BITS) <= 32, f"Packed representation needs {sum(_BITS)} bits — exceeds uint32."

# -- Output directory ----------------------------------------------------------
SSD_MOUNT  = "/media/hojunlee/Extreme Pro"
OUTPUT_DIR = os.path.join(SSD_MOUNT, "collision_configs")

# -- Batch / memory parameters -------------------------------------------------
BATCH_SIZE    = 128
PT_CHUNK_SIZE = 600_000
SAVE_EVERY    = 50
N_GPUS        = 2
N_batches     = (N_total + BATCH_SIZE - 1) // BATCH_SIZE


# =============================================================================
# WORKER
# =============================================================================

def worker(rank: int) -> None:
    """
    One process per GPU.  rank=0 → cuda:0, rank=1 → cuda:1.
    Processes batches  [rank * half_batches, (rank+1) * half_batches).
    Writes chunk files named  chunk_g{rank}_b{batch_idx:08d}.npy
    and a per-GPU checkpoint   checkpoint_g{rank}.json
    """
    d = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # ---- per-worker batch range ---------------------------------------------
    half      = (N_batches + N_GPUS - 1) // N_GPUS
    b_start_g = rank * half                          # global batch index start
    b_end_g   = min(b_start_g + half, N_batches)    # global batch index end (exclusive)
    n_my_batches = b_end_g - b_start_g

    # ---- initialise GPU resources -------------------------------------------
    chain = pk.build_serial_chain_from_urdf(
        open(f"{root_path}/{robot_name}.urdf").read(), "link6")
    chain = chain.to(device=d)
    s = pv.RobotSDF(chain, path_prefix=root_path,
                    link_sdf_cls=pv.cache_link_sdf_factory(
                        resolution=0.01, padding=1.0, device=d))

    pts       = PTS_CPU.to(d)
    grids_gpu = [g.to(d) for g in grids]

    enc_lo   = _ENC_LO_CPU.to(d)
    enc_step = _ENC_STEP_CPU.to(d)

    # ---- helpers (device-local closures) ------------------------------------
    def encode_batch_gpu(jc: torch.Tensor) -> np.ndarray:
        idx    = ((jc[:, :N_PACKED] - enc_lo) / enc_step).round().to(torch.int32)
        packed = torch.zeros(jc.shape[0], dtype=torch.int32, device=d)
        for i in range(N_PACKED):
            packed |= idx[:, i] << _ENC_SHIFTS[i]
        return packed.cpu().numpy().view(np.uint32)

    def generate_batch(flat_start: int, flat_end: int) -> torch.Tensor:
        flat_idx = torch.arange(flat_start, flat_end, dtype=torch.long, device=d)
        configs  = torch.empty(flat_idx.shape[0], N_joints, device=d)
        for i in range(N_joints):
            joint_idx     = (flat_idx // strides_int[i]) % N_per_joint_int[i]
            configs[:, i] = grids_gpu[i][joint_idx]
        return configs

    # ---- per-GPU checkpoint / output paths ----------------------------------
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, f"checkpoint_g{rank}.json")

    def flush_to_npy(live_buf: defaultdict, batch_idx: int, total_pairs: int) -> None:
        t0 = time.perf_counter()
        n_pts = n_rows = 0
        chunk: dict[int, np.ndarray] = {}
        for pt_idx, arrays in live_buf.items():
            if not arrays:
                continue
            data = np.concatenate(arrays, axis=0)
            chunk[pt_idx] = data
            n_pts  += 1
            n_rows += data.shape[0]
        live_buf.clear()

        if chunk:
            # GPU rank embedded in filename → no collision between workers
            fname = f"chunk_g{rank}_b{batch_idx:08d}.npy"
            final = os.path.join(OUTPUT_DIR, fname)
            tmp   = os.path.join(OUTPUT_DIR, f"chunk_g{rank}_b{batch_idx:08d}.tmp.npy")
            np.save(tmp, chunk)
            os.replace(tmp, final)

        tmp = CHECKPOINT_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "last_flushed_batch"   : batch_idx,
                "total_collision_pairs": total_pairs,
            }, f, indent=2)
        os.replace(tmp, CHECKPOINT_FILE)

        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"\n[GPU {rank}][flush] batch={batch_idx:,} | {n_pts:,} pts | "
            f"{n_rows:,} rows | took {elapsed:.1f} ms",
            flush=True,
        )

    def load_checkpoint() -> tuple[int, int]:
        if not os.path.exists(CHECKPOINT_FILE):
            print(f"[GPU {rank}] No checkpoint — starting from batch {b_start_g}.")
            return b_start_g - 1, 0
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        last  = ckpt["last_flushed_batch"]
        pairs = ckpt.get("total_collision_pairs", 0)
        print(
            f"[GPU {rank}] Checkpoint found — resuming from batch {last + 1:,} "
            f"({pairs:,} collision pairs so far)."
        )
        return last, pairs

    # ---- resume logic -------------------------------------------------------
    last_flushed_batch, total_collision_pairs = load_checkpoint()
    start_batch_idx = max(last_flushed_batch + 1, b_start_g)

    if start_batch_idx >= b_end_g:
        print(f"[GPU {rank}] All batches already completed.")
        return

    # ---- warmup -------------------------------------------------------------
    print(f"[GPU {rank}] Warming up on {d} ...")
    s.set_joint_configuration(generate_batch(b_start_g * BATCH_SIZE,
                                             b_start_g * BATCH_SIZE + 1))
    s(pts[:100])
    torch.cuda.synchronize(d)

    # ---- profiling state ----------------------------------------------------
    class RunningStats:
        def __init__(self): self.n = 0; self.mean = 0.0; self.M2 = 0.0
        def update(self, x: float):
            self.n += 1; delta = x - self.mean; self.mean += delta / self.n
            self.M2 += delta * (x - self.mean)
        @property
        def std(self): return (self.M2 / self.n) ** 0.5 if self.n > 1 else 0.0

    stats_gen    = RunningStats()
    stats_set_jc = RunningStats()
    stats_query  = RunningStats()
    stats_total  = RunningStats()
    PROFILE_EVERY = 100

    collision_buffer: defaultdict[int, list[np.ndarray]] = defaultdict(list)

    # ---- progress bar (one per worker, positional so they don't overlap) ----
    pbar = tqdm(
        total=n_my_batches,
        initial=start_batch_idx - b_start_g,
        desc=f"GPU {rank}",
        unit="batch",
        position=rank,
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]  {postfix}"
        ),
    )

    # ---- main loop ----------------------------------------------------------
    try:
        for batch_idx in range(start_batch_idx, b_end_g):
            bs = batch_idx * BATCH_SIZE
            be = min(bs + BATCH_SIZE, N_total)
            do_profile = (batch_idx % PROFILE_EVERY == 0)

            if do_profile: torch.cuda.synchronize(d); t0 = time.perf_counter()
            jc = generate_batch(bs, be)
            B  = jc.shape[0]
            if do_profile: torch.cuda.synchronize(d); t_gen = (time.perf_counter() - t0) * 1000

            if do_profile: torch.cuda.synchronize(d); t1 = time.perf_counter()
            s.set_joint_configuration(jc)
            if do_profile: torch.cuda.synchronize(d); t_set_jc = (time.perf_counter() - t1) * 1000

            if do_profile: torch.cuda.synchronize(d); t2 = time.perf_counter()

            n_collision_pairs = 0
            jc_packed = None

            for pt_start in range(0, N_pts, PT_CHUNK_SIZE):
                pt_end       = min(pt_start + PT_CHUNK_SIZE, N_pts)
                sdf_chunk, _ = s(pts[pt_start:pt_end])

                assert sdf_chunk.shape == (B, pt_end - pt_start), \
                    f"Unexpected sdf chunk shape {tuple(sdf_chunk.shape)}"

                mask_chunk = sdf_chunk <= 0
                n_pairs    = mask_chunk.sum().item()
                n_collision_pairs += n_pairs

                if n_pairs > 0:
                    if jc_packed is None:
                        jc_packed = encode_batch_gpu(jc)
                    mask_cpu = mask_chunk.cpu()
                    hit_pts  = mask_cpu.any(dim=0).nonzero(as_tuple=False).squeeze(1)
                    for local_idx in hit_pts.tolist():
                        global_idx      = pt_start + local_idx
                        hit_cfg_indices = mask_cpu[:, local_idx].nonzero(as_tuple=False).squeeze(1)
                        collision_buffer[global_idx].append(jc_packed[hit_cfg_indices.numpy()])

                del sdf_chunk, mask_chunk

            if do_profile: torch.cuda.synchronize(d); t_query = (time.perf_counter() - t2) * 1000

            total_collision_pairs += n_collision_pairs

            if do_profile:
                t_total = t_gen + t_set_jc + t_query
                stats_gen.update(t_gen); stats_set_jc.update(t_set_jc)
                stats_query.update(t_query); stats_total.update(t_total)
                pbar.set_postfix(ordered_dict={
                    "gen ms" : f"{stats_gen.mean:6.1f}",
                    "fk ms"  : f"{stats_set_jc.mean:6.1f}",
                    "sdf ms" : f"{stats_query.mean:6.1f}",
                    "tot ms" : f"{stats_total.mean:6.1f}",
                    "coll"   : f"{n_collision_pairs:,}",
                })
            pbar.update(1)

            if (batch_idx + 1) % SAVE_EVERY == 0 or batch_idx + 1 == b_end_g:
                flush_to_npy(collision_buffer, batch_idx, total_collision_pairs)

            del jc

        pbar.close()

    finally:
        if any(v for v in collision_buffer.values()):
            flush_to_npy(collision_buffer, batch_idx, total_collision_pairs)

    # ---- per-worker summary -------------------------------------------------
    print(f"\n[GPU {rank}] " + "-" * 80)
    print(f"[GPU {rank}]  {'':>6}  {'gen (ms)':>9}  {'set_jc (ms)':>12}  "
          f"{'sdf+buf (ms)':>13}  {'total (ms)':>11}")
    print(f"[GPU {rank}]  {'mean':>6}  {stats_gen.mean:>9.2f}  {stats_set_jc.mean:>12.2f}  "
          f"{stats_query.mean:>13.2f}  {stats_total.mean:>11.2f}")
    print(f"[GPU {rank}]  {'std':>6}  {stats_gen.std:>9.2f}  {stats_set_jc.std:>12.2f}  "
          f"{stats_query.std:>13.2f}  {stats_total.std:>11.2f}")
    projected = stats_total.mean * n_my_batches / 1000
    print(f"[GPU {rank}]  Projected ({n_my_batches:,} batches): "
          f"{projected:.1f} s / {projected/3600:.2f} hrs")
    print(f"[GPU {rank}]  Total collision pairs: {total_collision_pairs:,}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if not os.path.isdir(SSD_MOUNT):
        raise RuntimeError(
            f"SSD mount point '{SSD_MOUNT}' not found. "
            "Check the drive is mounted (run `lsblk` or `df -h`) and update SSD_MOUNT."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_available = torch.cuda.device_count()
    if n_available < N_GPUS:
        raise RuntimeError(
            f"Requested {N_GPUS} GPUs but only {n_available} are available."
        )

    print(f"N_pts         : {N_pts:,}")
    print(f"Total configs : {N_total:,}  (memory if materialised: {N_total*6*4/1e9:.1f} GB)")
    print(f"N_batches     : {N_batches:,}  (split ~{N_batches//N_GPUS:,} per GPU)")
    print(f"Encoding      : {_BITS} bits per joint  ({sum(_BITS)}/32 bits used)")
    print(f"Output dir    : {OUTPUT_DIR}")
    print(f"\nLaunching {N_GPUS} worker processes ...\n")

    # 'spawn' is required for CUDA — never use 'fork' with CUDA
    mp.set_start_method("spawn", force=True)
    processes = [mp.Process(target=worker, args=(rank,)) for rank in range(N_GPUS)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # ---- final summary -------------------------------------------------------
    total_chunks = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("chunk_g") and f.endswith(".npy")
    ]
    print(f"\nAll workers finished.  Chunk files in '{OUTPUT_DIR}': {len(total_chunks):,}")
    print("Run combine_npy.py to merge all chunks into one .npy file per grid point.")
