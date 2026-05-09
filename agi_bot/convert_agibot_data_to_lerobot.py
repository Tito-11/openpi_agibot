"""
Script for converting Agibot data to LeRobot format.
"""

import shutil
import os
import json
from pathlib import Path

import numpy as np
import cv2  # We use cv2 instead of imageio since we confirmed it's available
from scipy.spatial.transform import Rotation as R
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_NAME = "agibot_routeB"  # Name of the output dataset

def read_video(video_path):
    """
    Helper function to read a video file into a numpy array of RGB frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads in BGR format, so we convert it to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames)


def compute_valid_indices(actions, threshold=1e-3, min_idle_len=7):
    """
    通过计算平移量的变化来过滤静止帧 (Idle frames)。
    返回一个 boolean array，True 表示保留该帧。
    """
    if len(actions) <= 1:
        return np.ones(len(actions), dtype=bool)
        
    # 计算相邻两帧的位置差异 (只看前3维 x,y,z)
    pos_diffs = np.abs(actions[1:, :3] - actions[:-1, :3])
    
    # 判断每一帧是否是静止的
    is_idle = np.all(pos_diffs < threshold, axis=1)
    # 补齐第一帧
    is_idle = np.concatenate([[False], is_idle])
    
    keep_mask = np.ones(len(actions), dtype=bool)
    
    # 寻找连续静止的区间，如果长度大于 min_idle_len，则标记为 False (丢弃)
    idle_count = 0
    for i in range(len(is_idle)):
        if is_idle[i]:
            idle_count += 1
        else:
            if idle_count > min_idle_len:
                # 剔除这整段静止区间
                keep_mask[i - idle_count: i] = False
            idle_count = 0
            
    # 处理结尾可能存在的静止段
    if idle_count > min_idle_len:
        keep_mask[len(is_idle) - idle_count:] = False
        
    return keep_mask


def convert_to_6d(data_array):
    """
    Convert (N, 8) array with quaternions to (N, 10) array with 6D rotations.
    Original: [x, y, z, qw, qx, qy, qz, gripper]
    New: [x, y, z, r1, r2, r3, r4, r5, r6, gripper]
    """
    pos = data_array[:, :3]
    gripper = data_array[:, 7:8]
    
    # Scipy expects [x, y, z, w], so we reorder from [w, x, y, z]
    qw, qx, qy, qz = data_array[:, 3], data_array[:, 4], data_array[:, 5], data_array[:, 6]
    quats_scipy = np.stack([qx, qy, qz, qw], axis=1)
    
    rot_matrix = R.from_quat(quats_scipy).as_matrix() # (N, 3, 3)
    rot_6d = rot_matrix[:, :, :2].reshape(-1, 6) # (N, 6)
    
    return np.concatenate([pos, rot_6d, gripper], axis=-1)

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # Our exploration showed: actions (N, 8), states (N, 8), video resolution (240, 320, 3)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="agibot",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    data_dir_path = Path(data_dir)
    # Find all episode directories
    episodes = sorted([d for d in data_dir_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    for ep_dir in episodes:
        print(f"Processing {ep_dir.name}...")
        
        try:
            actions_raw = np.load(ep_dir / "actions.npy")
            states_raw = np.load(ep_dir / "states.npy")
            
            # Convert Quaternions to 6D Rotation representation
            actions = convert_to_6d(actions_raw)
            states = convert_to_6d(states_raw)
            
            # Read videos
            # Based on meta.json: "head_color" (camera_0) and "hand_left" (camera_1)
            head_vid_path = ep_dir / "camera_0.mp4"
            hand_vid_path = ep_dir / "camera_1.mp4"
            
            head_frames = read_video(head_vid_path)
            hand_frames = read_video(hand_vid_path)
        except Exception as e:
            print(f"Skipping episode {ep_dir.name} due to data loading issue: {e}")
            continue
            
        # Match lengths (in case video dropped frames at the end)
        num_steps = min(len(actions), len(states), len(head_frames), len(hand_frames))
        
        # --- 数据清洗：计算 Idle 帧掩码 ---
        # 使用过滤脚本过滤掉长时间的停顿
        keep_mask = compute_valid_indices(actions[:num_steps])
        valid_steps = np.sum(keep_mask)
        print(f"  Filtering: kept {valid_steps}/{num_steps} frames (removed {num_steps - valid_steps} idle frames).")
        
        if valid_steps < 10:
            print(f"  Episode {ep_dir.name} has too few valid frames. Skipping.")
            continue
            
        frame_idx = 0
        for i in range(num_steps):
            if not keep_mask[i]:
                continue
                
            dataset.add_frame(
                {
                    "image": head_frames[i],
                    "wrist_image": hand_frames[i],
                    "state": states[i].astype(np.float32),
                    "actions": actions[i].astype(np.float32),
                    "task": "cartesian grasp target",
                }
            )
            frame_idx += 1
            
        dataset.save_episode()
        print(f"  Saved {frame_idx} valid frames for {ep_dir.name}.")

    dataset.consolidate()
    print(f"\n✅ Dataset successfully converted and saved to: {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["agibot", "rlds", "lerobot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
