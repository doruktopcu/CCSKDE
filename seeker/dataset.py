import json
import os
import re
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.data_utils import normalize_pose, shanghaitech_hr_skip
from utils.sequence_utils import clip_to_segments


class SkeletonSequenceDataset(Dataset):
    def __init__(self, path_to_json_dir, path_to_vid_dir=None, evaluate=False, filter_conf=0, **dataset_args):
        super().__init__()
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        self.path_to_vid_dir = path_to_vid_dir
        self.eval = evaluate
        if evaluate:
            filter_conf = 0
        self.train_seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)

    
        self.segs_data_np, self.segs_meta, self.person_keys, self.global_data_np, \
        self.global_data, self.segs_score_np = gen_dataset(path_to_json_dir, filter_conf=filter_conf, **dataset_args)
        self.segs_meta = np.array(self.segs_meta)
        
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    def __getitem__(self, index):
        sample_index = index
        data_transformed = np.array(self.segs_data_np[index])

        data_transformed = normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...],
                                              **self.args).squeeze(axis=0).transpose(2, 0, 1)

        ret_arr = [data_transformed]

        ret_arr += [self.segs_score_np[sample_index]]
        return ret_arr

    def get_all_data(self):
        segs_data_np = normalize_pose(self.segs_data_np.transpose((0, 2, 3, 1)), **self.args).transpose(
                (0, 3, 1, 2))
        return list(segs_data_np)

    def __len__(self):
        return self.num_samples


def get_dataset_and_loader(args, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset_args = {'seg_len': args.seg_len, "dataset": args.dataset, 'train_seg_conf_th': args.train_seg_conf_th}
    dataset, loader = dict(), dict()
    if only_test:
        splits = ['test']
    elif args.dataset == 'UBnormal':
        splits = ['train', 'validate', 'test'] 
    else:    
        splits = ['train', 'test']
    for split in splits:
        evaluate = split == 'test'
        dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set
        dataset_args['vid_path'] = args.vid_path[split]
        dataset_args['split'] = split
        dataset[split] = SkeletonSequenceDataset(args.pose_path[split], path_to_vid_dir=args.vid_path[split],
                                        evaluate=evaluate,
                                        filter_conf=args.filter_conf,
                                        **dataset_args)
        
        if split != 'train':
            l= {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
            loader[split] = DataLoader(dataset[split], **l, shuffle=(split == 'train'))
        else:
            loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    if only_test:
        loader['train'] = None
        loader['validate'] = loader['test']
        dataset['validate'] = dataset['test']
    if args.dataset != 'UBnormal':
        dataset['validate'] = dataset['test']
        loader['validate'] = loader['test']
    return dataset, loader



def gen_dataset(person_json_root, filter_conf=None,**dataset_args):
    segs_data_np = []
    segs_score_np = []
    segs_meta = []
    global_data = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 24)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    dataset = dataset_args.get('dataset', 'ShanghaiTech')

    dir_list = os.listdir(person_json_root)
    json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
    for person_dict_fn in tqdm(json_list):
        if dataset == "UBnormal":
            type, scene_id, clip_id = \
                re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*', person_dict_fn)[0]
            clip_id = type + "_" + clip_id
        elif dataset == "avenue":
            clip_id = person_dict_fn[:2]
            scene_id = clip_id
        elif dataset == "MSAD":
            scene_id, clip_id = re.match(r"([a-zA-Z-_]+)(\d+)", person_dict_fn.replace("_alphapose_tracked_person.json", '')).groups()
        else:
            scene_id, clip_id = person_dict_fn.split('_')[:2]
            if shanghaitech_hr_skip(dataset=="ShaghaiTech-HR", scene_id, clip_id):
                continue
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys, single_pos_np, _, score_segs_data_np = clip_to_segments(
            clip_dict, start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            dataset=dataset,
            filter_conf=filter_conf)

        _, _, _, global_data_np, global_data, _ = clip_to_segments(clip_dict, start_ofst, 1, 1, scene_id=scene_id,
                                                                       clip_id=clip_id,
                                                                       global_pose_data=global_data,
                                                                       dataset=dataset, filter_conf=filter_conf)
        segs_data_np.append(clip_segs_data_np)
        segs_score_np.append(score_segs_data_np)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}

    # Global data
    global_data_np = np.expand_dims(np.concatenate(global_data, axis=0), axis=1)
    segs_data_np = np.concatenate(segs_data_np, axis=0)
    segs_score_np = np.concatenate(segs_score_np, axis=0)
    
    # to coco18 format
    segs_data_np = keypoints17_to_coco18(segs_data_np)
    global_data_np = keypoints17_to_coco18(global_data_np)
    global_data = [keypoints17_to_coco18(data) for data in global_data]

    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    global_data_np = np.transpose(global_data_np, (0, 3, 1, 2)).astype(np.float32)

    if seg_conf_th > 0.0:
        segs_data_np, segs_meta, segs_score_np = \
            seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th)
    
    return segs_data_np, segs_meta, person_keys, global_data_np, global_data, segs_score_np




def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int64)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
    sum_confs = segs_score_np.mean(axis=1)
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    return seg_data_filt, seg_meta_filt, segs_score_np
