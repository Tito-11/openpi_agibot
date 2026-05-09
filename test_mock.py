import numpy as np
from pathlib import Path

def compute_valid_indices(actions, threshold=1e-3, min_idle_len=7):
    if len(actions) <= 1:
        return np.ones(len(actions), dtype=bool)
    pos_diffs = np.abs(actions[1:, :3] - actions[:-1, :3])
    is_idle = np.all(pos_diffs < threshold, axis=1)
    is_idle = np.concatenate([[False], is_idle])
    keep_mask = np.ones(len(actions), dtype=bool)
    idle_count = 0
    for i in range(len(is_idle)):
        if is_idle[i]:
            idle_count += 1
        else:
            if idle_count > min_idle_len:
                keep_mask[i - idle_count: i] = False
            idle_count = 0
    if idle_count > min_idle_len:
        keep_mask[len(is_idle) - idle_count:] = False
    return keep_mask

def convert_to_6d(data_array):
    pos = data_array[:, :3]
    gripper = data_array[:, 7:8]
    qw, qx, qy, qz = data_array[:, 3], data_array[:, 4], data_array[:, 5], data_array[:, 6]
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    wx2, wy2, wz2 = qw * x2, qw * y2, qw * z2
    xx2, xy2, xz2 = qx * x2, qx * y2, qx * z2
    yy2, yz2, zz2 = qy * y2, qy * z2, qz * z2
    r00, r01, r02 = 1.0 - (yy2 + zz2), xy2 - wz2, xz2 + wy2
    r10, r11, r12 = xy2 + wz2, 1.0 - (xx2 + zz2), yz2 - wx2
    rot_6d = np.stack([r00, r01, r02, r10, r11, r12], axis=1)
    return np.concatenate([pos, rot_6d, gripper], axis=-1)

ep_dir = Path("agi_bot/data/cartesian_grasp_routeB_5pt_clean_v2/episode_0001")
if ep_dir.exists():
    actions_raw = np.load(ep_dir / "actions.npy")
    print(f"Original actions shape: {actions_raw.shape}")
    actions = convert_to_6d(actions_raw)
    print(f"6D actions shape: {actions.shape}")
    keep_mask = compute_valid_indices(actions)
    valid_steps = np.sum(keep_mask)
    print(f"Filtered {valid_steps} valid frames out of {len(actions)}, removing {len(actions) - valid_steps} idle frames")
    print(f"First valid Action Sample 10D: {np.round(actions[keep_mask][0], 3)}")
else:
    print("Data not found")
