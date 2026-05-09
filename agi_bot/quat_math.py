import numpy as np

def quat_to_rot_matrix(quats):
    """
    Convert quaternions [x, y, z, w] to rotation matrices.
    """
    x = quats[:, 0]
    y = quats[:, 1]
    z = quats[:, 2]
    w = quats[:, 3]

    x2 = x + x
    y2 = y + y
    z2 = z + z

    wx2 = w * x2
    wy2 = w * y2
    wz2 = w * z2
    xx2 = x * x2
    xy2 = x * y2
    xz2 = x * z2
    yy2 = y * y2
    yz2 = y * z2
    zz2 = z * z2

    rot = np.empty((quats.shape[0], 3, 3))
    rot[:, 0, 0] = 1.0 - (yy2 + zz2)
    rot[:, 0, 1] = xy2 - wz2
    rot[:, 0, 2] = xz2 + wy2

    rot[:, 1, 0] = xy2 + wz2
    rot[:, 1, 1] = 1.0 - (xx2 + zz2)
    rot[:, 1, 2] = yz2 - wx2

    rot[:, 2, 0] = xz2 - wy2
    rot[:, 2, 1] = yz2 + wx2
    rot[:, 2, 2] = 1.0 - (xx2 + yy2)

    return rot
