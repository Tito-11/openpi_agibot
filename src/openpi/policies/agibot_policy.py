import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model as _model

@dataclasses.dataclass(frozen=True)
class AgibotInputs(transforms.DataTransformFn):
    """
    Transforms for Agibot input data.
    """
    def __call__(self, data: dict) -> dict:
        in_images = data.get("images", {})
        
        base_image = in_images.get("head_color")
        if base_image is None:
            raise ValueError("Expected 'head_color' in images dict")
            
        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": in_images.get("hand_left_color", np.zeros_like(base_image)),
            "right_wrist_0_rgb": np.zeros_like(base_image)
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_ if "hand_left_color" in in_images else np.False_,
            "right_wrist_0_rgb": np.False_
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        return inputs

@dataclasses.dataclass(frozen=True)
class AgibotOutputs(transforms.DataTransformFn):
    """
    Transforms for Agibot output data.
    Takes the raw model output and slices it down to the configured Agibot action space.
    """
    action_dim: int = 10

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][..., : self.action_dim])
        return {"actions": actions}
