import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model as _model

@dataclasses.dataclass(frozen=True)
class AgibotInputs(transforms.DataTransformFn):
    """
    Transforms for Agibot input data (10-dim with 6D rotation).
    Ensures that the gripper (10th dimension, index 9) is strictly within [0.0, 1.0].
    """
    def __call__(self, data: dict) -> dict:
        # Clip gripper state and action to [0, 1]
        # actions shape: (horizon, 10), state shape: (10,)
        
        # State: [3 pos, 6 rot, 1 gripper]
        if "state" in data:
            state = np.asarray(data["state"])
            state[9] = np.clip(state[9], 0.0, 1.0)
            data["state"] = state
            
        # Actions: [horizon, 3 pos, 6 rot, 1 gripper]
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions[..., 9] = np.clip(actions[..., 9], 0.0, 1.0)
            data["actions"] = actions
            
        return data

@dataclasses.dataclass(frozen=True)
class AgibotOutputs(transforms.DataTransformFn):
    """
    Transforms for Agibot output data.
    Takes the 32-dim raw model output and slices it down to the 10-dim Agibot action space.
    The continuous gripper prediction is thresholded into 0 or 1.
    """
    def __call__(self, data: dict) -> dict:
        # The model outputs a 32-dim array by default, but our action space is 10.
        actions = np.asarray(data["actions"][..., :10])
        
        # Binarize gripper output (0 or 1) using 0.5 as threshold
        # Since Flow Matching generates continuous outputs, we must hard-threshold the discrete gripper.
        actions[..., 9] = np.where(actions[..., 9] > 0.5, 1.0, 0.0)
        
        return {"actions": actions}
