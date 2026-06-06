import numpy as np


def clip_to_segments(clip_dict, start_ofst=0, seg_stride=1, seg_len=12, scene_id='', clip_id='',
                         global_pose_data=[], dataset="ShanghaiTech", filter_conf=None):
    """
    Generate an array of segmented sequences, each object is a segment and a corresponding metadata array
    """
    pose_segs_data = []
    score_segs_data = []
    pose_segs_meta = []
    person_keys = {}
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = single_pose_dict2np(clip_dict, idx)
        if dataset == 'UBnormal':
            key = ('{:02d}_{}_{:02d}'.format(int(scene_id), clip_id, int(idx)))
        elif dataset == 'avenue':
            key = ('{:02d}'.format(int(clip_id)))
        elif dataset == 'MSAD':
            key = ('{}_{:04d}_{:02d}'.format(scene_id, int(clip_id), int(idx)))
        else:
            key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))
        person_keys[key] = sing_pose_keys
        curr_pose_segs_np, curr_pose_segs_meta, curr_pose_score_np = split_person_sequence_to_segments(sing_pose_np,
                                                                                            sing_pose_meta,
                                                                                            sing_pose_keys,
                                                                                            start_ofst, seg_stride,
                                                                                            seg_len,
                                                                                            scene_id=scene_id,
                                                                                            clip_id=clip_id,
                                                                                            single_score_np=sing_scores_np,
                                                                                            dataset=dataset, filter_conf=filter_conf)
        if curr_pose_segs_np.shape[0] == 0:
            continue
        pose_segs_data.append(curr_pose_segs_np)
        score_segs_data.append(curr_pose_score_np)
        if sing_pose_np.shape[0] > seg_len:
            global_pose_data.append(sing_pose_np)
        pose_segs_meta += curr_pose_segs_meta
    if len(pose_segs_data) == 0:
        pose_segs_data_np = np.empty(0).reshape(0, seg_len, 17, 3)
        score_segs_data_np = np.empty(0).reshape(0, seg_len)
    else:
        pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)
        score_segs_data_np = np.concatenate(score_segs_data, axis=0)
    global_pose_data_np = np.concatenate(global_pose_data, axis=0)
    del pose_segs_data
    return pose_segs_data_np, pose_segs_meta, person_keys, global_pose_data_np, global_pose_data, score_segs_data_np


def single_pose_dict2np(person_dict, idx):
    single_person = person_dict[str(idx)]
    sing_pose_np = []
    sing_scores_np = []
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    single_person_dict_keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [index, first_frame]
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
        sing_pose_np.append(curr_pose_np)
        sing_scores_np.append(single_person[key]['scores'])
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    sing_scores_np = np.stack(sing_scores_np, axis=0)
    return sing_pose_np, sing_pose_meta, single_person_dict_keys, sing_scores_np


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def impute_missing_frames(single_pose_np, single_pose_keys, seg_len):

    diff = np.diff(single_pose_keys)
    missing_indices = np.where(diff > 1)[0]

    splits_keys = []
    splits_pose = []
    l = 0
    for i in missing_indices:
        splits_keys.append(single_pose_keys[l:i+1])
        splits_pose.append(single_pose_np[l:i+1, ...])

        # expand 
        length = diff[i]-1
        last = single_pose_keys[i]

        if length > seg_len:
            l = i + 1
            continue
        else:
            splits_keys.append(np.arange(last+1, last+length+1))
            a = single_pose_np[i, ...]
            b = single_pose_np[i+1, ...]
            interp_poses = np.array([a + (b - a) * t / (length + 1) for t in range(1, length + 1)])
            splits_pose.append(interp_poses)
            l = i + 1

    splits_keys.append(single_pose_keys[i+1:])
    splits_pose.append(single_pose_np[i+1:, ...])

    return np.concatenate(splits_pose), np.concatenate(splits_keys).astype(int).tolist()


def split_person_sequence_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12,
                           scene_id='', clip_id='', single_score_np=None, dataset="ShanghaiTech", filter_conf=None):
    single_pose_keys = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    if filter_conf:
        certain_frames = single_pose_np.mean(axis=1)[:, 2] > filter_conf
        single_pose_np = single_pose_np[certain_frames]
        single_pose_keys = np.array(single_pose_keys)[certain_frames].tolist()
    diff = np.diff(single_pose_keys)
    if not np.all(diff == 1):
        single_pose_np, single_pose_keys = impute_missing_frames(single_pose_np, single_pose_keys, seg_len)
        single_score_np = single_pose_np[..., -1][:, 0]
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_score_np = np.empty([0, seg_len])
    pose_segs_meta = []
    num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(np.int64)
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys[start_ind]
        if is_seg_continuous(single_pose_keys, start_key, seg_len):
            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            curr_score = single_score_np[start_ind:start_ind + seg_len].reshape(1, seg_len)
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_score_np = np.append(pose_score_np, curr_score, axis=0)
            if dataset == "UBnormal":
                pose_segs_meta.append([int(scene_id), clip_id, int(single_pose_meta[0]), int(start_key)])
            elif dataset == "MSAD":
                pose_segs_meta.append([scene_id, int(clip_id), int(single_pose_meta[0]), int(start_key)])
            else:
                pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])
    return pose_segs_np, pose_segs_meta, pose_score_np

