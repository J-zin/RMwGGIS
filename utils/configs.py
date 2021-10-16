import os
import torch
import argparse
import pickle as cp


cmd_opt = argparse.ArgumentParser(description='Argparser for debm', allow_abbrev=False)
cmd_opt.add_argument('-save_dir', default='./results', help='result output root')
cmd_opt.add_argument('-data', default='2spirals', help='data name')
cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')

cmd_opt.add_argument('-phase', default='train', help='train/test')
cmd_opt.add_argument('-energy_model_dump', default=None, help='load energy model dump')
cmd_opt.add_argument('-epoch_load', default=None, type=int, help='epoch for loading')

cmd_opt.add_argument('-embed_dim', default=256, type=int, help='embed size')
cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='gradient clip')
cmd_opt.add_argument('-weight_clip', default=-1, type=float, help='weight clip')
cmd_opt.add_argument('-f_lr_scale', default=0.2, type=float, help='scale of f learning rate')

cmd_opt.add_argument('-num_epochs', default=1000, type=int, help='num epochs')
cmd_opt.add_argument('-batch_size', default=128, type=int, help='batch size')
cmd_opt.add_argument('-gibbs_rounds', default=5, type=int, help='rounds of gibbs sampling')
cmd_opt.add_argument('-num_importance_samples', default=10, type=int, help='number of importance samples')

cmd_opt.add_argument('-epoch_save', default=100, type=int, help='num epochs between save')
cmd_opt.add_argument('-iter_per_epoch', default=100, type=int, help='num iterations per epoch')

# synthetic data
cmd_opt.add_argument('-discrete_dim', default=32, type=int, help='embed size')
cmd_opt.add_argument('-binmode', default='gray', help='normal/rand/gray')

cmd_opt.add_argument('-int_scale', default=None, type=float, help='scale for float2int')
cmd_opt.add_argument('-plot_size', default=None, type=float, help='scale for plot')

cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    cmd_args.save_dir = os.path.join(cmd_args.save_dir, cmd_args.data)
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

if cmd_args.epoch_load is not None:
    cmd_args.energy_model_dump = os.path.join(cmd_args.save_dir, 'energy_func-%d.ckpt' % cmd_args.epoch_load)

def set_device(gpu):
    if torch.cuda.is_available() and gpu >= 0:
        cmd_args.gpu = gpu
        cmd_args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        cmd_args.gpu = -1
        cmd_args.device = torch.device('cpu')
        print('use cpu')