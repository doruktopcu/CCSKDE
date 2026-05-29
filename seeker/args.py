import argparse
import os
import time


def init_args():
    parser = init_parser()
    args = parser.parse_args()
    return init_sub_args(args)


def init_sub_args(args):
    if args.dataset == "avenue":
        args.vid_path = {
            "train": os.path.join(args.data_dir, args.dataset, "train/images/"),
            "validate": os.path.join(args.data_dir, args.dataset, "validation/frames"),
            "test": os.path.join(args.data_dir, args.dataset, "test/frames/"),
        }

        args.pose_path = {
            "train": os.path.join(args.data_dir, args.dataset, "train/"),
            "validate": os.path.join(args.data_dir, args.dataset, "validation/"),
            "test": os.path.join(args.data_dir, args.dataset, "test/"),
        }

    else:
        dataset = "UBnormal" if args.dataset == "UBnormal" else "ShanghaiTech"
        if (
            args.vid_path_train
            and args.vid_path_test
            and args.pose_path_train
            and args.pose_path_test
        ):
            args.vid_path = {
                "train": args.vid_path_train,
                "validate": args.vid_path_validate,
                "test": args.vid_path_test,
            }

            args.pose_path = {
                "train": args.pose_path_train,
                "test": args.pose_path_test,
                "validate": args.pose_path_validate,
            }
        else:
            args.vid_path = {
                "train": os.path.join(args.data_dir, dataset, "train/images/"),
                "validate": os.path.join(args.data_dir, dataset, "validation/frames"),
                "test": os.path.join(args.data_dir, dataset, "test/frames/"),
            }

            args.pose_path = {
                "train": os.path.join(args.data_dir, dataset, "pose", "train/"),
                "validate": os.path.join(args.data_dir, dataset, "pose", "validation/"),
                "test": os.path.join(args.data_dir, dataset, "pose", "test/"),
            }
        if args.dataset == "MSAD":
            args.pose_path = {
                "train": os.path.join(args.data_dir, "pose", "train/"),
                "validate": os.path.join(args.data_dir, "pose", "validation/"),
                "test": os.path.join(args.data_dir, "pose", "test/"),
            }
    args.pose_path["train_abnormal"] = args.pose_path_train_abnormal
    args.ckpt_dir = None
    model_args = args_rm_prefix(args, "model_")
    return args, model_args


def init_parser(default_data_dir="./data", default_exp_dir="./exp_dir"):
    parser = argparse.ArgumentParser(prog="SeeKer")
    # General Args
    parser.add_argument(
        "--vid_path_train", type=str, default=None, help="Path to training vids"
    )
    parser.add_argument(
        "--pose_path_train_abnormal",
        type=str,
        default=None,
        help="Path to training vids",
    )
    parser.add_argument(
        "--pose_path_train", type=str, default=None, help="Path to training pose"
    )
    parser.add_argument(
        "--vid_path_validate", type=str, default=None, help="Path to validation vids"
    )
    parser.add_argument(
        "--pose_path_validate", type=str, default=None, help="Path to validation pose"
    )
    parser.add_argument(
        "--vid_path_test", type=str, default=None, help="Path to test vids"
    )
    parser.add_argument(
        "--pose_path_test", type=str, default=None, help="Path to test pose"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="UBnormal",
        choices=["ShanghaiTech", "ShanghaiTech-HR", "UBnormal", "avenue", "MSAD"],
        help="Dataset for Eval",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        metavar="DEV",
        help="Device for feature calculation (default: 'cuda:0')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="S",
        default=999,
        help="Random seed, use 999 for random (default: 999)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Path to directory holding .npy and .pkl files (default: {})".format(
            default_data_dir
        ),
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=default_exp_dir,
        # metavar="EXP_DIR",
        help="Path to the directory where models will be saved (default: {})".format(
            default_exp_dir
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        metavar="W",
        help="number of dataloader workers (0=current thread) (default: 32)",
    )

    # Data Params
    parser.add_argument(
        "--droppout",
        type=float,
        default=0,
        help="Droppout.",
    )
    parser.add_argument(
        "--train_seg_conf_th",
        "-th",
        type=float,
        default=0.0,
        metavar="CONF_TH",
        help="Training set threshold Parameter (default: 0.0)",
    )
    parser.add_argument(
        "--seg_len",
        type=int,
        default=24,
        metavar="SGLEN",
        help="Number of frames for training segment sliding window, a multiply of 6 (default: 12)",
    )
    parser.add_argument(
        "--seg_stride",
        type=int,
        default=1,
        metavar="SGST",
        help="Stride for training segment sliding window",
    )
    parser.add_argument(
        "--filter_conf",
        type=float,
        default=0,
        help="confidence threshold",
    )

    # Model Params
    parser.add_argument(
        "--checkpoint", type=str, metavar="model", help="Path to a pretrained model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, metavar="B", help="Batch size for train"
    )
    parser.add_argument(
        "--sigma", type=int, default=0, metavar="sigma", help="Sigma for smoothing"
    )
    parser.add_argument(
        "--epochs",
        "-model_e",
        type=int,
        default=10,
        metavar="E",
        help="Number of epochs per cycle",
    )
    parser.add_argument(
        "--model_optimizer",
        "-model_o",
        type=str,
        default="adam",
        metavar="model_OPT",
        help="Optimizer",
    )
    parser.add_argument(
        "--model_lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="Optimizer Learning Rate Parameter",
    )

    #autoreg model params
    parser.add_argument(
        "--n_layers", type=int, default=1, help="Number of layers."
    )

    parser.add_argument(
        "--expansion_factor", type=int, default=1, help="expansion factor"
    )

    return parser


def args_rm_prefix(args, prefix):
    wp_args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    wp_args_dict = vars(wp_args)
    for key, value in args_dict.items():
        if key.startswith(prefix):
            model_key = key[len(prefix) :]
            wp_args_dict[model_key] = value

    return wp_args


def create_exp_dirs(experiment_dir, dirmap=""):
    time_str = time.strftime("%b%d_%H%M")

    experiment_dir = os.path.join(experiment_dir, dirmap, time_str)
    dirs = [experiment_dir]

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created")
        return experiment_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)


