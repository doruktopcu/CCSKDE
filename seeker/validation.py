import os
import numpy as np
from tqdm import tqdm
import re
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


from utils.scoring_utils import fill_and_smooth_fw
from utils.data_utils import shanghaitech_hr_skip

def score_anomalies(scores_kp, metadata, args=None, split='test', ret_gt=False, sigma=0):
    gt_arr, scores_arr = get_all_frame_scores(scores_kp, metadata, args=args, split=split, sigma=sigma)
    scores_arr = smooth_scores(scores_arr, 2)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)

    auc = score_auc(scores_np, gt_np)
    if ret_gt:
        return auc, scores_np, gt_np
    return auc


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max() +1
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()-1
    auc = roc_auc_score(gt, scores_np)

    fpr, tpr, thresholds = roc_curve(gt, scores_np)

    target_tpr = 0.95
    idx = (tpr >= target_tpr).argmax()
    fpr_at_target_tpr = fpr[idx]
    threshold_at_target_tpr = thresholds[idx]
    print("FPR@95: ", fpr_at_target_tpr)

    # calculate average precision
    avg_precision = average_precision_score(gt, scores_np)
    print("Average Precision: ", avg_precision)

    # print("Threshold@TPR=95%: ", threshold_at_target_tpr)
    return auc



def get_all_frame_scores(scores, metadata, args=None, split='test', sigma=0):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)

    if args.dataset == 'UBnormal':
        pose_segs_root = f'data/UBnormal/pose/{split}'
        clip_list = os.listdir(pose_segs_root)
        clip_list = sorted(
            fn.replace("alphapose_tracked_person.json", "tracks.txt") for fn in clip_list if fn.endswith('.json'))
        per_frame_scores_root = os.path.join(args.data_dir, f'UBnormal/gt/{split}/')
    elif args.dataset == 'avenue':
        per_frame_scores_root = f'/mnt/sdb1/adelic/datasets/avenue/ground_truth_demo/testing_label_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.mat'))
    elif args.dataset == 'MSAD':
        per_frame_scores_root = f'/mnt/sdb1/datasets/MSAD_blur/results/pose/gt/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    else:
        per_frame_scores_root = os.path.join(args.data_dir, 'ShanghaiTech/gt/test_frame_mask/')
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))

    print("Scoring {} clips".format(len(clip_list)))
    for clip in tqdm(clip_list):
        clip_gt, clip_score = get_scores_in_clip(scores, clip, metadata_np, metadata, per_frame_scores_root, args, sigma=sigma)
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)

    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()+1
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()-1

    index = 0
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index]
            index += 1

    return dataset_gt_arr, dataset_scores_arr



def get_scores_in_clip(scores, clip, metadata_np, metadata, per_frame_scores_root, args, sigma=0):
    if args.dataset == 'UBnormal':
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
        clip_id = type + "_" + clip_id
    elif args.dataset == 'avenue':
        clip_id = int(clip.split('_')[0])
        scene_id = clip_id
    elif args.dataset == 'MSAD':
        scene_id, clip_id = re.match(r"([a-zA-Z-_]+)(\d+)", clip.replace(".npy", '')).groups()
    else:
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
        if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
            return None, None
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    if args.dataset == 'avenue':
        clip_dict = loadmat(clip_res_fn)
        gt = clip_dict['volLabel'][0]
        clip_gt = np.zeros_like(gt, dtype=np.int32)
        for t in range(gt.shape[0]):
            if np.any(gt[t] == 1):
                clip_gt[t] = 1
    else:
        clip_gt = np.load(clip_res_fn)

    scores_zeros = np.ones(clip_gt.shape[0]) * (- np.inf)
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    kp_scores_zeros = np.ones((clip_gt.shape[0], 18)) * (- np.inf)
    if len(clip_fig_idxs) == 0:
        clip_kp_person_scores_dict = {0: np.copy(kp_scores_zeros)}
    else:
        clip_kp_person_scores_dict = {i: np.copy(kp_scores_zeros) for i in clip_fig_idxs}
        
    final_idx = clip_gt.shape[0]-1
    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds]

        if pid_scores.shape[0] < args.seg_len:
            continue
        s_t_ = pid_scores.reshape((pid_scores.shape[0], args.seg_len, 18))[:, -1, :]
        s_t = pid_scores.reshape((pid_scores.shape[0], args.seg_len, 18)).sum(-1)
        # frames where person is present in video
        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int) # first frames of segments

        s, pid_frame_inds_ = fill_and_smooth_fw(s_t, pid_frame_inds, args, sigma=sigma, final_idx=final_idx)

        clip_person_scores_dict[person_id][pid_frame_inds_] = s
        clip_kp_person_scores_dict[person_id][pid_frame_inds] = s_t_

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amax(clip_ppl_score_arr, axis=0)

    scores_zeros = np.ones(clip_gt.shape[0]) * (- np.inf)
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
    return clip_gt, clip_score




def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr





