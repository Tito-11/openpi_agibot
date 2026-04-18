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
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
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
    
    actions = np.load(ep_dir / "actions.npy")
    states = np.load(ep_dir / "states.npy")
    head_frames = read_video(ep_dir / "camera_0.mp4")
    hand_frames = read_video(ep_dir / "camera_1.mp4")
        
    num_steps = min(len(actions), len(states), len(head_frames), len(hand_frames))
    print(f"总计提取出 {num_steps} 帧有效数据，开始装载至 Parquet 表中...")

    for i in range(num_steps):
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
