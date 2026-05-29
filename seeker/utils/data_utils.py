import math

import numpy as np
import torch


def get_aff_trans_mat(sx=1, sy=1, tx=0, ty=0, rot=0, shearx=0., sheary=0., flip=False):
    """
    Generate affine transfomation matrix (torch.tensor type) for transforming pose sequences
    :rot is given in degrees
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(3, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=torch.float32)
    shear_mat = torch.tensor([[1, shearx, 0], [sheary, 1, 0], [0, 0, 1]], dtype=torch.float32)
    rot_mat = torch.tensor([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32)
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(shear_mat, aff_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat


def apply_pose_transform(pose, trans_mat):
    """ Given a set of pose sequences of shape (Channels, Time_steps, Vertices, M[=num of figures])
    return its transformed form of the same sequence. 3 Channels are assumed (x, y, conf) """

    # We isolate the confidence vector, replace with ones, than plug back after transformation is done
    conf = np.expand_dims(pose[2], axis=0)
    ones_vec = np.ones_like(conf)
    pose_w_ones = np.concatenate([pose[:2], ones_vec], axis=0)
    if len(pose.shape) == 3:
        einsum_str = 'ktv,ck->ctv'
    else:
        einsum_str = 'ktvm,ck->ctvm'
    pose_transformed_wo_conf = np.einsum(einsum_str, pose_w_ones, trans_mat)
    pose_transformed = np.concatenate([pose_transformed_wo_conf[:2], conf], axis=0)
    return pose_transformed


class PoseTransform(object):
    """ A general class for applying transformations to pose sequences, empty init returns identity """

    def __init__(self, sx=1, sy=1, tx=0, ty=0, rot=0, shearx=0., sheary=0., flip=False, trans_mat=None):
        """ An explicit matrix overrides all parameters"""
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = get_aff_trans_mat(sx, sy, tx, ty, rot, shearx, sheary, flip)

    def __call__(self, x):
        x = apply_pose_transform(x, self.trans_mat)
        return x



def normalize_pose(pose_data, **kwargs):
    pose_data_zero_mean = pose_data.copy()
    pose_data_zero_mean[..., :2] = (pose_data[..., :2] - pose_data[..., :2].mean(axis=(1, 2))[:, None, None, :]) / pose_data[..., 1].std(axis=(1, 2))[:, None, None, None]
    return pose_data_zero_mean



SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]
def shanghaitech_hr_skip(shanghaitech_hr, scene_id, clip_id):
    if not shanghaitech_hr:
        return shanghaitech_hr
    if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
        return True
    return False


