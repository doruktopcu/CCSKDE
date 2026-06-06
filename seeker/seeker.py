import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader

from models.partial_autoregressive import PartialAutoregressiveFC
from training import SeeKerTrainer
from utils.training_utils import init_optimizer


def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.seed == 999:  
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    
    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, only_test=(pretrained is not None))

    in_dim = 2*18*args.seg_len
    model = PartialAutoregressiveFC(dim=in_dim, hidden_dims=args.n_layers*[args.expansion_factor*(in_dim-2)], droppout=args.droppout)
    print(model)
    model.to(args.device)
    writer = SummaryWriter()
    trainer = SeeKerTrainer(args, model, loader['train'], loader['test'], loader['validate'], 
                                           dataset['validate'].metadata, dataset['test'].metadata,
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      log_writer=writer, 
                      dataset=args.dataset)
    

    if not pretrained:
        print(args.ckpt_dir)
        auc_val = trainer.train()
    else:
        trainer.load_checkpoint(pretrained)
        auc_test = trainer.test()


if __name__ == '__main__':
    main()
