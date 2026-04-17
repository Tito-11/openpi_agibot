import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
repo_id = "test_agibot"
ds = LeRobotDataset.create(
    repo_id=repo_id,
    fps=10,
    robot_type="agibot",
    features={
        "image": {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
        "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
    }
)
print("Success:", ds.root)
