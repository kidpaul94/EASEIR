"""
GPU-Parallel Self-Collision Map Generator
==========================================
Uses pytorch_kinematics (FK) + pytorch_volumetric (per-link SDFs) to check
self-collision for large batches of joint configs in parallel on the GPU.

Algorithm per batch of B joint configs:
  1. FK → (B, 4, 4) transform per link
  2. For each non-adjacent link pair (A, B):
       a. Transform link A's surface points into link B's local frame
       b. Query link B's SDF                              → (B, N_pts)
       c. Flag configs where any SDF value ≤ SDF_THRESHOLD
  3. OR-reduce across all pairs → binary collision flag per config

Output:
  self_collision_map_gpu.npy  shape (N, 6)
  columns: [j1, j2, j3, j4, j5, is_collision]  angles in DEGREES
"""

import os
import xml.etree.ElementTree as ET
import argparse

import numpy as np
import torch
import pytorch_kinematics as pk
import pytorch_volumetric as pv
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

ROOT_PATH   = "/home/hojunlee/Work/EASEIR++/test"
URDF_PATH   = f"{ROOT_PATH}/m0609_wo_eef.urdf"
OUTPUT_PATH = "./self_collision_map_gpu.npy"
END_LINK    = "link6"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE     = 32000   # joint configs per GPU batch — tune to VRAM
N_SURF_PTS     = 128      # surface points sampled per link mesh
SDF_THRESHOLD  = 0.0      # SDF <= 0 → inside mesh → collision
SDF_RESOLUTION = 0.01    # voxel resolution in metres

# Joint limits (degrees) — joint 6 is held fixed at 0
JOINT_LIMITS_DEG = {
    0: np.arange(-180., 180., 5.0),   # Joint 1
    1: np.arange(-90., 95., 5.0),   # Joint 2
    2: np.arange(-145., 150., 5.0),   # Joint 3
    3: np.arange(-180., 180., 5.0),   # Joint 4
    4: np.arange(-180., 180., 5.0),   # Joint 5
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_chain(urdf_path: str) -> pk.SerialChain:
    chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), END_LINK)
    return chain.to(dtype=torch.float32, device=DEVICE)


def parse_meshes_from_urdf(urdf_path: str) -> dict:
    """
    Parse URDF collision tags.
    Returns dict: link_name -> (mesh_path, scale np.array(3,))
    """
    tree = ET.parse(urdf_path)
    mesh_map = {}
    for link_el in tree.getroot().findall("link"):
        link_name = link_el.get("name")
        coll = link_el.find("collision")
        if coll is None:
            continue
        geom = coll.find("geometry")
        if geom is None:
            continue
        mesh_el = geom.find("mesh")
        if mesh_el is None:
            continue
        filename  = mesh_el.get("filename")
        scale_str = mesh_el.get("scale", "1 1 1")
        scale     = np.array([float(s) for s in scale_str.split()])
        mesh_map[link_name] = (filename, scale)
    return mesh_map


def build_link_assets(urdf_path: str) -> tuple[dict, dict]:
    """
    For each link with a collision mesh build:
      - A CachedSDF (voxel grid) in the link's own local frame for fast GPU queries
      - N_SURF_PTS surface points sampled from the same mesh

    Correct pv API chain:
      mesh_path -> MeshObjectFactory -> MeshSDF (GT) -> CachedSDF (voxelised, fast)

    Returns:
      link_sdfs   : dict  link_name -> CachedSDF
      surface_pts : dict  link_name -> Tensor (N, 3) on DEVICE
    """
    mesh_map    = parse_meshes_from_urdf(urdf_path)
    link_sdfs   = {}
    surface_pts = {}

    for link_name, (mesh_path, scale) in mesh_map.items():
        if not os.path.exists(mesh_path):
            alt = os.path.join(os.path.dirname(urdf_path), os.path.basename(mesh_path))
            if os.path.exists(alt):
                mesh_path = alt
            else:
                print(f"  [skip] mesh not found: {mesh_path}")
                continue
        try:
            # scale must be scalar — URDF "0.001 0.001 0.001" is always uniform
            scale_val   = float(scale[0])
            obj_factory = pv.MeshObjectFactory(mesh_path, scale=scale_val)

            # MeshSDF: exact ray-traced GT SDF
            mesh_sdf = pv.MeshSDF(obj_factory)

            # CachedSDF: bake into a voxel grid for fast repeated GPU queries
            bb = obj_factory.bounding_box(padding=0.1)
            link_sdfs[link_name] = pv.CachedSDF(
                object_name=link_name,
                resolution=SDF_RESOLUTION,
                range_per_dim=bb,
                gt_sdf=mesh_sdf,
                device=DEVICE,
            )

            # Sample surface points from the open3d mesh inside obj_factory
            pcd = obj_factory._mesh.sample_points_uniformly(number_of_points=N_SURF_PTS)
            surface_pts[link_name] = torch.tensor(
                np.asarray(pcd.points), dtype=torch.float32, device=DEVICE
            )
            print(f"  [ok] {link_name}")

        except Exception as e:
            print(f"  [skip] {link_name}: {e}")

    print(f"\nPer-link SDFs built : {len(link_sdfs)}")
    print(f"Surface pts sampled : {len(surface_pts)}")
    return link_sdfs, surface_pts


def get_non_adjacent_pairs(chain: pk.SerialChain, available_links: set) -> list[tuple]:
    """
    Return (link_A, link_B) pairs that are both available AND non-adjacent
    in the kinematic chain (differ by more than 1 in chain order).
    """
    chain_links = [l for l in chain.get_link_names() if l in available_links]
    pairs = [
        (chain_links[i], chain_links[j])
        for i in range(len(chain_links))
        for j in range(i + 2, len(chain_links))   # i+1 is adjacent — skip it
    ]
    print(f"Non-adjacent pairs  : {len(pairs)}")
    for a, b in pairs:
        print(f"  {a}  <->  {b}")
    return pairs


def batch_fk(
    chain:          pk.SerialChain,
    configs_rad:    torch.Tensor,    # (B, n_active)
    n_chain_joints: int,
) -> dict:
    """
    Batched FK for B configs. Pads to n_chain_joints with zeros (joint 6 = 0).
    Returns dict: link_name -> (B, 4, 4) local-to-world transform.
    """
    B = configs_rad.shape[0]
    if configs_rad.shape[1] < n_chain_joints:
        pad = torch.zeros(B, n_chain_joints - configs_rad.shape[1], device=DEVICE)
        configs_rad = torch.cat([configs_rad, pad], dim=1)

    ret = chain.forward_kinematics(configs_rad, end_only=False)
    return {name: tf.get_matrix() for name, tf in ret.items()}


def check_batch(
    chain:          pk.SerialChain,
    link_sdfs:      dict,
    surface_pts:    dict,
    non_adj_pairs:  list[tuple],
    configs_rad:    torch.Tensor,    # (B, n_active)
    n_chain_joints: int,
) -> torch.Tensor:                   # (B,) bool
    """
    Returns True for configs where any non-adjacent link pair penetrates.
    """
    B         = configs_rad.shape[0]
    collision = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    T         = batch_fk(chain, configs_rad, n_chain_joints)

    for link_a, link_b in non_adj_pairs:
        if link_a not in surface_pts or link_b not in link_sdfs:
            continue
        if link_a not in T or link_b not in T:
            continue

        pts_a = surface_pts[link_a]     # (N, 3) in link_a local frame
        N     = pts_a.shape[0]

        # Transform: link_a local -> world -> link_b local
        T_b_a = torch.linalg.inv(T[link_b]) @ T[link_a]   # (B, 4, 4)

        # Homogeneous transform of surface points
        ones     = torch.ones(N, 1, device=DEVICE)
        pts_h    = torch.cat([pts_a, ones], dim=1)          # (N, 4)
        pts_in_b = (T_b_a @ pts_h.T).permute(0, 2, 1)[..., :3]  # (B, N, 3)

        # SDF query: (B*N, 3) -> (B*N,)
        sdf_vals, _ = link_sdfs[link_b](pts_in_b.reshape(B * N, 3))
        penetrating = (sdf_vals.reshape(B, N) <= SDF_THRESHOLD).any(dim=1)
        collision  |= penetrating

    return collision


def total_configs() -> int:
    return int(np.prod([len(v) for v in JOINT_LIMITS_DEG.values()]))


def joint_config_generator(batch_size: int):
    """
    Fully lazy joint config generator — only j3×j4×j5 (~77 MB) lives in RAM.
    Outer loops over j1 and j2 are pure Python iteration.

    Memory breakdown:
      inner comb (j3×j4×j5): 149×180×180 = 4.8M rows × 4 cols × 4B = ~77 MB
      one batch buffer      : batch_size  × 5 cols × 4B (e.g. 1k rows = 20 KB)

    Matches data_gen.py ordering: outer j2, inner cartesian(j1, j3, j4, j5).
    """
    j1 = torch.tensor(JOINT_LIMITS_DEG[0], dtype=torch.float32)
    j2 = torch.tensor(JOINT_LIMITS_DEG[1], dtype=torch.float32)
    j3 = torch.tensor(JOINT_LIMITS_DEG[2], dtype=torch.float32)
    j4 = torch.tensor(JOINT_LIMITS_DEG[3], dtype=torch.float32)
    j5 = torch.tensor(JOINT_LIMITS_DEG[4], dtype=torch.float32)

    # Smallest safe inner grid: j3 × j4 × j5 (~77 MB)
    inner = torch.stack(
        torch.meshgrid(j3, j4, j5, indexing='ij'), dim=-1
    ).reshape(-1, 3)   # (K, 3)  K = len(j3)*len(j4)*len(j5)
    K = len(inner)

    buf     = []
    buf_len = 0

    for j2_val in j2:
        for j1_val in j1:
            # One (j1, j2) slice: K rows, columns = [j1, j2, j3, j4, j5]
            row = torch.cat([
                j1_val.expand(K, 1),
                j2_val.expand(K, 1),
                inner,             # j3, j4, j5
            ], dim=1)              # (K, 5)

            buf.append(row)
            buf_len += K

            while buf_len >= batch_size:
                full    = torch.cat(buf, dim=0)
                batch   = full[:batch_size]
                rest    = full[batch_size:]
                buf     = [rest] if len(rest) > 0 else []
                buf_len = len(rest)
                yield batch, torch.deg2rad(batch)

    if buf_len > 0:
        batch = torch.cat(buf, dim=0)
        yield batch, torch.deg2rad(batch)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_collision_map(
    urdf_path:   str = URDF_PATH,
    output_path: str = OUTPUT_PATH,
    batch_size:  int = BATCH_SIZE,
) -> None:

    print(f"Device : {DEVICE}")
    if DEVICE.type == "cpu":
        print("  Warning: CUDA not available — running on CPU (will be slow).")

    chain          = build_chain(urdf_path)
    n_chain_joints = len(chain.get_joint_parameter_names())
    print(f"Revolute joints in chain: {n_chain_joints}\n")

    print("Building per-link SDFs and sampling surface points:")
    link_sdfs, surface_pts = build_link_assets(urdf_path)

    available     = set(link_sdfs.keys()) & set(surface_pts.keys())
    non_adj_pairs = get_non_adjacent_pairs(chain, available)

    N_total   = total_configs()
    n_batches = (N_total + batch_size - 1) // batch_size
    print(f"\nTotal configurations : {N_total:,}")
    print(f"Batch size           : {batch_size:,}")
    print(f"Number of batches    : {n_batches:,}")
    print(f"Output               : {output_path}\n")

    tmp_path        = output_path + ".tmp.npy"
    total_written   = 0
    total_collision = 0

    with open(tmp_path, "wb") as f:
        for configs_deg, configs_rad in tqdm(
            joint_config_generator(batch_size), total=n_batches, unit="batch"
        ):
            flags = check_batch(
                chain, link_sdfs, surface_pts, non_adj_pairs,
                configs_rad.to(DEVICE), n_chain_joints,
            ).cpu().numpy().astype(np.float32)

            total_written   += len(flags)
            total_collision += int(flags.sum())

            # Keep only colliding rows — no flag column needed since all rows collide
            colliding_batch = configs_deg.numpy()[flags.astype(bool)]
            if len(colliding_batch) > 0:
                np.save(f, colliding_batch)

    # Consolidate streamed collision-only batches into one .npy file
    print("\nConsolidating batches...")
    chunks = []
    with open(tmp_path, "rb") as f:
        while True:
            try:
                chunks.append(np.load(f))
            except Exception:
                break
    os.remove(tmp_path)

    if chunks:
        data = np.vstack(chunks)   # (N_colliding, 5): [j1, j2, j3, j4, j5]
    else:
        data = np.empty((0, 5), dtype=np.float32)
    np.save(output_path, data)

    print(f"Saved  -> {output_path}")
    print(f"Shape  : {data.shape}  (colliding configs only, no flag column)")
    print(f"Collision rate: {total_collision / max(total_written, 1):.2%}")


def load_and_query(map_path: str, query_angles_deg: np.ndarray) -> bool:
    """Return True if query config (degrees) is in the collision map (nearest grid point)."""
    data  = np.load(map_path)          # (N_colliding, 5): all rows are collisions
    dists = np.linalg.norm(data - query_angles_deg, axis=1)
    return bool(np.min(dists) < 5.0)   # threshold: within 5 deg = collision


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU self-collision map generator")
    parser.add_argument("--urdf",       default=URDF_PATH,   help="Path to robot URDF")
    parser.add_argument("--output",     default=OUTPUT_PATH, help="Output .npy path")
    parser.add_argument("--batch-size", default=BATCH_SIZE,  type=int,
                        help="Configs per GPU batch (tune to VRAM)")
    args = parser.parse_args()

    generate_collision_map(
        urdf_path=args.urdf,
        output_path=args.output,
        batch_size=args.batch_size,
    )