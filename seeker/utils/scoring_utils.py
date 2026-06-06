from scipy.ndimage import gaussian_filter1d
import numpy as np

def smooth_score(scores, sigma=7):
    for sig in range(1, sigma):
        scores = gaussian_filter1d(scores, sigma=sig)
    return scores

def _is_consecutive(person_metadata_inds):
    sorted_inds = np.sort(person_metadata_inds)
    return np.all(np.diff(sorted_inds) == 1)


def fill_and_smooth_fw(s_t, pid_frame_inds, args, sigma=1, final_idx=None):
    if _is_consecutive(pid_frame_inds):
        pid_frame_inds_ = np.concatenate([pid_frame_inds, np.arange(pid_frame_inds[-1]+1, pid_frame_inds[-1]+args.seg_len+1)])
        # expand the first segment to the left
        first = s_t[0, :]
        
        first = np.ones_like(first) * s_t[:, -1][0]

        s = np.concatenate([first, s_t[:,-1]], axis=0)
        s = smooth_score(s, sigma=sigma)
    else:
        _scores = s_t[:,-1]

        # Compute the difference between consecutive elements
        diff = np.diff(pid_frame_inds)
        
        # Find the indices where the difference is not equal to 1
        split_indices = np.where(diff != 1)[0]

        splits_ind = []
        splits_scores = []
        l = 0
        for i in split_indices:
            splits_ind.append(pid_frame_inds[l:i+1])
            splits_scores.append(_scores[l:i+1])
            l = i + 1
        splits_ind.append(pid_frame_inds[i+1:])
        splits_scores.append(_scores[i+1:])

        # expand first segments to the left
        splits_ind_ = []
        splits_scores_ = []

        for i in range(0, len(splits_ind)):
            ind = splits_ind[i]
            s = splits_scores[i]

            length = diff[split_indices[i-1]]-1

            if length < args.seg_len:
                splits_ind_.append(np.concatenate((ind, np.arange(ind[-1]+1, (ind[-1]+args.seg_len+1))[-length:])))
            else:
                length = args.seg_len
                splits_ind_.append(np.concatenate((ind, np.arange(ind[-1]+1, ind[-1]+args.seg_len+1))))
            
            sc = np.concatenate((np.ones((length, )) * s[0], s))
            splits_scores_.append(
                smooth_score(sc, sigma=sigma)
            )    

        s = np.concatenate(splits_scores_)
        pid_frame_inds_ = np.concatenate(splits_ind_)

        s = s[pid_frame_inds_ <= final_idx]
        pid_frame_inds_ = pid_frame_inds_[pid_frame_inds_ <= final_idx]

    return s, pid_frame_inds_



