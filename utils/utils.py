import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sympy.combinatorics.graycode import GrayCode

from utils.configs import cmd_args
from utils.plot_2d import plot_samples
from model.sampler import GibbsSampler
from dataset.dataset import OnlineToyDataset

def setup_data(args):
    bm, inv_bm = get_binmap(args.discrete_dim, args.binmode)
    db = OnlineToyDataset(args.data)
    if args.int_scale is None:
        args.int_scale = db.int_scale
    else:   
        db.int_scale = args.int_scale
    if args.plot_size is None:
        args.plot_size = db.f_scale
    else:
        db.f_scale = args.plot_size
    return db, bm, inv_bm

def get_binmap(discrete_dim, binmode):
    b = discrete_dim // 2 - 1
    all_bins = []
    for i in range(1 << b):
        bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
        all_bins.append('0' + bx)
        all_bins.append('1' + bx)
    vals = all_bins[:]
    if binmode == 'rand':
        print('remapping binary repr with random permute')
        random.shuffle(vals)
    elif binmode == 'gray':
        print('remapping binary repr with gray code')
        a = GrayCode(b)
        vals = []
        for x in a.generate_gray():
            vals.append('0' + x)
            vals.append('1' + x)
    else:
        assert binmode == 'normal'
    bm = {}
    inv_bm = {}
    for i, key in enumerate(all_bins):
        bm[key] = vals[i]
        inv_bm[vals[i]] = key
    return bm, inv_bm

def float2bin(samples, bm):
    bin_list = []
    for i in range(samples.shape[0]):
        x, y = samples[i] * cmd_args.int_scale
        bx, by = compress(x), compress(y)
        bx, by = bm[bx], bm[by]
        bin_list.append(np.array(list(bx + by), dtype=int))
    return np.array(bin_list)

def bin2float(samples, inv_bm):
    floats = []
    for i in range(samples.shape[0]):
        s = ''
        for j in range(samples.shape[1]):
            s += str(samples[i, j])
        x, y = s[:cmd_args.discrete_dim//2], s[cmd_args.discrete_dim//2:]
        x, y = inv_bm[x], inv_bm[y]
        x, y = recover(x), recover(y)
        x /= cmd_args.int_scale
        y /= cmd_args.int_scale
        floats.append((x, y))
    return np.array(floats)

def compress(x):
    bx = np.binary_repr(int(abs(x)), width=cmd_args.discrete_dim // 2 - 1)
    bx = '0' + bx if x >= 0 else '1' + bx
    return bx

def recover(bx):
    x = int(bx[1:], 2)
    return x if bx[0] == '0' else -x

def plot_heat(score_func, bm, out_file=None):
    w = 100
    size = cmd_args.plot_size
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])
    heat_samples = float2bin(np.concatenate((xx, yy), axis=-1), bm)
    heat_samples = torch.from_numpy(heat_samples).to(cmd_args.device)
    heat_score = F.softmax(-score_func(heat_samples).view(1, -1), dim=-1)
    a = heat_score.view(w, w).data.cpu().numpy()
    a = np.flip(a, axis=0)
    plt.imshow(a)
    plt.axis('equal')
    plt.axis('off')
    if out_file is None:
        out_file = os.path.join(cmd_args.save_dir, 'heat.pdf')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def plot_sampler(score_func, inv_bm, out_file=None):
    gibbs_sampler = GibbsSampler(2, cmd_args.discrete_dim, cmd_args.device)

    samples = []
    for _ in tqdm(range(50)):
        with torch.no_grad():
            init_samples = gibbs_sampler(score_func, cmd_args.gibbs_rounds, num_samples=cmd_args.batch_size)
        samples.append(bin2float(init_samples.data.cpu().numpy(), inv_bm))
    samples = np.concatenate(samples, axis=0)
    plot_samples(samples, out_file, lim=4.1, axis=False)

def plot_data(db, out_file=None):
    data = db.gen_batch(128 * 50)
    plot_samples(data, out_file, lim=4.1, axis=False)
