from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
d = LeRobotDataset.create(repo_id="test", features={"image": {"dtype": "image", "shape": (240, 320, 3)}, "wrist_image": {"dtype": "image", "shape": (240, 320, 3)}, "state": {"dtype": "float32", "shape": (8,)}, "action": {"dtype": "float32", "shape": (8,)}})
print(d.features)
