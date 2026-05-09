"""
测试脚本：仅转换 Episode_0001 数据到本地，用于展示和验证 LeRobot 格式转换结果。
"""

import shutil
import os
from pathlib import Path

# 在导入 LeRobot 之前，强制修改环境变量中的 Hugging Face 数据集缓存路径，指向当前的本地测试目录
local_test_dir = Path(__file__).parent.absolute() / "test_converted_data"
os.environ["HF_LEROBOT_HOME"] = str(local_test_dir)

import numpy as np
import cv2  

import lerobot.common.datasets.lerobot_dataset as lerobot_ds
# 双重确保全局变量已被修改为本地目录
lerobot_ds.HF_LEROBOT_HOME = local_test_dir

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


REPO_NAME = "test_episode_0001" 

def read_video(video_path):
    """读取视频文件为 RGB numpy 数组"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
    """
    Convert (N, 8) array with quaternions to (N, 10) array with 6D rotations.
    Original: [x, y, z, qw, qx, qy, qz, gripper]
    New: [x, y, z, r1, r2, r3, r4, r5, r6, gripper]
    """
    pos = data_array[:, :3]
    gripper = data_array[:, 7:8]
    
    # Scipy expects [x, y, z, w], so we reorder from [qw, qx, qy, qz] -> [qx, qy, qz, qw]
    qw, qx, qy, qz = data_array[:, 3], data_array[:, 4], data_array[:, 5], data_array[:, 6]
    
    # Numpy implementation to replace scipy.spatial.transform.Rotation
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    wx2, wy2, wz2 = qw * x2, qw * y2, qw * z2
    xx2, xy2, xz2 = qx * x2, qx * y2, qx * z2
    yy2, yz2, zz2 = qy * y2, qy * z2, qz * z2

    r00, r01, r02 = 1.0 - (yy2 + zz2), xy2 - wz2, xz2 + wy2
    r10, r11, r12 = xy2 + wz2, 1.0 - (xx2 + zz2), yz2 - wx2

    rot_6d = np.stack([r00, r01, r02, r10, r11, r12], axis=1) # (N, 6)
    
    return np.concatenate([pos, rot_6d, gripper], axis=-1)


def main():
    output_path = local_test_dir / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    print(f"[{REPO_NAME}] 初始化 LeRobotDataset，保存路径: {output_path}")

    # 创建 LeRobotDataset
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
        image_writer_threads=2,
        image_writer_processes=0, # 对于单个 episode 测试，关闭多进程避免部分环境阻塞
    )

    # 仅锁准 episode_0001
    ep_dir = Path("agi_bot/data/cartesian_grasp_routeB_5pt_clean_v2/episode_0001")
    if not ep_dir.exists():
        print(f"❌ 找不到对应的数据目录: {ep_dir.absolute()}")
        return

    print(f"🔄 正在处理单集数据 {ep_dir.name}...")
    
    actions_raw = np.load(ep_dir / "actions.npy")
    states_raw = np.load(ep_dir / "states.npy")
    
    # 将四元数转为 6D
    actions = convert_to_6d(actions_raw)
    states = convert_to_6d(states_raw)
    
    head_frames = read_video(ep_dir / "camera_0.mp4")
    hand_frames = read_video(ep_dir / "camera_1.mp4")
        
    num_steps = min(len(actions), len(states), len(head_frames), len(hand_frames))
    
    # 过滤 Idle 帧
    keep_mask = compute_valid_indices(actions[:num_steps])
    valid_steps = np.sum(keep_mask)
    print(f"提取出 {num_steps} 帧数据，通过 Idle 过滤保留 {valid_steps} 帧有效数据，开始装载...")

    for i in range(num_steps):
        if not keep_mask[i]:
            continue
            
        dataset.add_frame(
            {
                "image": head_frames[i],
                "wrist_image": hand_frames[i],
                "state": states[i].astype(np.float32),
                "actions": actions[i].astype(np.float32),
                "task": "grasp_bottle",
            }
        )
    dataset.save_episode()
    
    # 部分版本的 LeRobotDataset 可能提供 consolidate 方法强制刷新元数据
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()

    print(f"\n✅ 转换完成！验证数据格式请查看: {output_path}")
    print("在此目录中，您应当能看到 Parquet 数据表格、视频/图片导出档以及 meta 描述信息。")

if __name__ == "__main__":
    main()
