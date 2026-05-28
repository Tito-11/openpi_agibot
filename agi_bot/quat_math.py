import numpy as np


def reorder_quat_xyzw(quat: np.ndarray, input_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    if input_order == "xyzw":
        return quat.copy()
    if input_order == "wxyz":
        return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
    raise ValueError(f"Unsupported quaternion order: {input_order}")


def quat_xyzw_to_output(quat_xyzw: np.ndarray, output_order: str) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    if quat_xyzw.ndim == 1:
        quat_xyzw = quat_xyzw[None, :]
    if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
        raise ValueError(f"Expected quaternion array shape [N, 4], got {quat_xyzw.shape}")
    if output_order == "xyzw":
        return quat_xyzw.copy()
    if output_order == "wxyz":
        return np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=-1)
    raise ValueError(f"Unsupported quaternion order: {output_order}")


def rot_matrix_to_rot6d_interleaved(rot_matrices: np.ndarray) -> np.ndarray:
    rot_matrices = np.asarray(rot_matrices, dtype=np.float32)
    if rot_matrices.ndim == 2:
        rot_matrices = rot_matrices[None, ...]
    if rot_matrices.ndim != 3 or rot_matrices.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotation matrices shape [N, 3, 3], got {rot_matrices.shape}")
    return rot_matrices[:, :, :2].reshape(-1, 6)


def rot6d_interleaved_to_rot_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.ndim == 1:
        rot6d = rot6d[None, :]
    if rot6d.ndim != 2 or rot6d.shape[1] != 6:
        raise ValueError(f"Expected rot6d shape [N, 6], got {rot6d.shape}")

    first_two_cols = rot6d.reshape(-1, 3, 2)
    col0 = first_two_cols[:, :, 0]
    col1 = first_two_cols[:, :, 1]

    col0 = col0 / np.clip(np.linalg.norm(col0, axis=1, keepdims=True), 1e-8, None)
    col1 = col1 - np.sum(col0 * col1, axis=1, keepdims=True) * col0
    col1 = col1 / np.clip(np.linalg.norm(col1, axis=1, keepdims=True), 1e-8, None)
    col2 = np.cross(col0, col1)

    return np.stack([col0, col1, col2], axis=-1).astype(np.float32)
