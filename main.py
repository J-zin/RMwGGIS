import torch
import random
import numpy as np

from model.train import main_loop
from utils.utils import setup_data
from utils.configs import cmd_args, set_device
from model.f_energy import MLPScore

if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    db, bm, inv_bm = setup_data(cmd_args)

    energy_func = MLPScore(cmd_args.discrete_dim, [cmd_args.embed_dim] * 3 + [1]).to(cmd_args.device)
    main_loop(db, bm, inv_bm, energy_func)