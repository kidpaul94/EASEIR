"""
Visualize Self-Collision Configurations using Open3D
=====================================================
Loads colliding joint configurations from self_collide_gpu.py output and
renders each pose sequentially in an Open3D window.

FK is computed via pytorch_kinematics (same as the collision map generator).
Each link mesh is loaded once, then repositioned each frame by applying the
FK transform relative to the previous pose.

Usage:
    python visualize_collisions.py
    python visualize_collisions.py --urdf ./test/m0609_wo_eef.urdf
                                   --map  ./self_collision_map_gpu.npy
                                   --max  20
                                   --delay 1.5
"""

import argparse
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import open3d as o3d
import pytorch_kinematics as pk

# ── Config ────────────────────────────────────────────────────────────────────

URDF_PATH = "/home/hojunlee/Work/EASEIR++/test/m0609_wo_eef.urdf"
MAP_PATH  = "./self_collision_map_gpu.npy.tmp.npy"
END_LINK  = "link6"
MAX_CONFIGS = 20
DELAY       = 5.0    # seconds between configs

# Map columns [j1..j5] -> revolute joint names in chain order. Joint 6 = 0.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4_0", "joint5"]

# One distinct colour per link (RGB, 0-1)
LINK_COLORS = [
    [0.8, 0.2, 0.2],   # base_link  - red
    [0.2, 0.6, 0.2],   # link1      - green
    [0.2, 0.2, 0.8],   # link2_0    - blue
    [0.8, 0.6, 0.1],   # link2_1    - orange
    [0.6, 0.2, 0.8],   # link2_2    - purple
    [0.1, 0.7, 0.7],   # link3      - cyan
    [0.9, 0.4, 0.4],   # link4_0    - salmon
    [0.4, 0.8, 0.4],   # link4_1    - light green
    [0.4, 0.4, 0.9],   # link5      - light blue
    [0.9, 0.9, 0.2],   # link6      - yellow
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── URDF mesh parsing ─────────────────────────────────────────────────────────

def parse_visual_meshes(urdf_path: str) -> dict:
    """
    Returns dict: link_name -> (mesh_path, scale np.array(3,))
    Reads <visual> tags for nicer rendering, falls back to <collision>.
    """
    tree = ET.parse(urdf_path)
    mesh_map = {}
    for link_el in tree.getroot().findall("link"):
        link_name = link_el.get("name")
        for tag in ("visual", "collision"):
            el = link_el.find(tag)
            if el is None:
                continue
            geom = el.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is None:
                continue
            filename  = mesh_el.get("filename")
            scale_str = mesh_el.get("scale", "1 1 1")
            scale     = np.array([float(s) for s in scale_str.split()])
            mesh_map[link_name] = (filename, scale)
            break
    return mesh_map


# ── FK ────────────────────────────────────────────────────────────────────────

def build_chain(urdf_path: str) -> pk.SerialChain:
    chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), END_LINK)
    return chain.to(dtype=torch.float32, device=DEVICE)


def get_link_transforms(chain: pk.SerialChain, angles_deg: np.ndarray) -> dict:
    """
    Run FK for a single config (degrees).
    Returns dict: link_name -> (4, 4) numpy world transform.
    """
    n_joints   = len(chain.get_joint_parameter_names())
    angles_rad = np.deg2rad(angles_deg).tolist()
    angles_rad += [0.0] * (n_joints - len(angles_rad))   # pad joint 6 with 0

    cfg = torch.tensor([angles_rad], dtype=torch.float32, device=DEVICE)
    ret = chain.forward_kinematics(cfg, end_only=False)
    return {name: tf.get_matrix()[0].cpu().numpy() for name, tf in ret.items()}


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_meshes(mesh_map: dict) -> dict:
    """
    Load each link mesh into open3d, apply scale, paint a unique colour.
    Returns dict: link_name -> o3d.geometry.TriangleMesh (in local link frame).
    """
    meshes = {}
    for idx, (link_name, (mesh_path, scale)) in enumerate(mesh_map.items()):
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if not mesh.has_vertices():
                print(f"  [skip] empty mesh : {link_name}")
                continue
            mesh.scale(float(scale[0]), center=np.zeros(3))
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(LINK_COLORS[idx % len(LINK_COLORS)])
            meshes[link_name] = mesh
            print(f"  [ok] {link_name}")
        except Exception as e:
            print(f"  [skip] {link_name}: {e}")
    return meshes


# ── Collision map ─────────────────────────────────────────────────────────────

def load_collision_configs(map_path: str) -> np.ndarray:
    data      = np.load(map_path)
    colliding = data[data[:, 5] == 1]
    print(f"Loaded map    : {map_path}")
    print(f"Total configs : {len(data):,}")
    print(f"Colliding     : {len(colliding):,}  ({100*len(colliding)/len(data):.2f}%)")
    return colliding[:, :5]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(urdf_path: str, map_path: str, max_configs: int, delay: float) -> None:
    print(f"Device : {DEVICE}\n")

    chain     = build_chain(urdf_path)
    colliding = load_collision_configs(map_path)
    n_show    = min(max_configs, len(colliding))

    print("\nLoading meshes:")
    mesh_map = parse_visual_meshes(urdf_path)
    meshes   = load_meshes(mesh_map)

    if not meshes:
        print("No meshes loaded — check mesh paths in URDF.")
        return

    # Apply zero-config FK transform to each mesh as starting pose
    T_current = get_link_transforms(chain, np.zeros(5))
    for link_name, mesh in meshes.items():
        if link_name in T_current:
            mesh.transform(T_current[link_name])

    # Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Self-Collision Visualizer", width=1024, height=768)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().light_on            = True

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    for mesh in meshes.values():
        vis.add_geometry(mesh)

    vis.poll_events()
    vis.update_renderer()

    print(f"\nShowing {n_show} colliding configurations...\n")

    for i in range(n_show):
        angles_deg = colliding[i]
        label = "  ".join(
            f"j{j+1}={angles_deg[j]:>7.2f}°" for j in range(len(JOINT_NAMES))
        )
        print(f"[{i+1:>3}/{n_show}]  {label}")

        T_new = get_link_transforms(chain, angles_deg)

        # Reposition each mesh: undo old transform, apply new
        for link_name, mesh in meshes.items():
            if link_name in T_new and link_name in T_current:
                mesh.transform(np.linalg.inv(T_current[link_name]))
                mesh.transform(T_new[link_name])
                vis.update_geometry(mesh)

        T_current = T_new

        vis.poll_events()
        vis.update_renderer()
        time.sleep(delay)

    print("\nDone. Close the window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize self-collision configs via Open3D"
    )
    parser.add_argument("--urdf",  default=URDF_PATH,   help="Path to URDF file")
    parser.add_argument("--map",   default=MAP_PATH,    help="Path to .npy collision map")
    parser.add_argument("--max",   default=MAX_CONFIGS, type=int,
                        help="Max colliding configs to show")
    parser.add_argument("--delay", default=DELAY,       type=float,
                        help="Seconds between configs")
    args = parser.parse_args()

    main(
        urdf_path=args.urdf,
        map_path=args.map,
        max_configs=args.max,
        delay=args.delay,
    )