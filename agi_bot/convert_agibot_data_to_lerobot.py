"""
Script for converting Agibot data to LeRobot format.
"""

import shutil
import os
import json
from pathlib import Path

import numpy as np
import cv2  # We use cv2 instead of imageio since we confirmed it's available
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
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
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
            actions = np.load(ep_dir / "actions.npy")
            states = np.load(ep_dir / "states.npy")
            
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
