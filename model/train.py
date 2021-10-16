import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from utils.configs import cmd_args
from utils.utils import float2bin, plot_heat, plot_sampler, plot_data

def score_func(samples, energy_func):
    x = samples.clone().type(torch.FloatTensor).requires_grad_().to(samples.device)
    energy = energy_func(x)
    score = torch.autograd.grad(energy.sum(), x, retain_graph=True, create_graph=True)[0]
    return score

def get_proposal_dist(samples, energy_func):
    score = score_func(samples, energy_func)
    numerator = 2 * (2 * samples - 1) * score
    dominator = torch.logsumexp(numerator, dim=-1, keepdim=True)
    log_proposal_dit = numerator - dominator
    proposal_dit = torch.softmax(log_proposal_dit, dim=-1)
    return proposal_dit.detach()

def importance_sampling(pos_samples, prop_dit):
    n, dim = pos_samples.shape
    is_num = cmd_args.num_importance_samples

    torch_cat_dit = torch.distributions.Categorical(prop_dit)
    is_idx = torch_cat_dit.sample([is_num]).transpose(0, 1)
    prop_val = prop_dit[torch.arange(is_idx.size(0)).unsqueeze(1), is_idx]

    neg_samples = []
    for i in range(n):      # Todo: without a for loop
        idx = is_idx[i]
        neg = pos_samples[i].unsqueeze(0).expand(is_num, dim)
        c = torch.zeros((is_num, dim)).to(pos_samples.device)
        c[torch.arange(c.size(0)).unsqueeze(1), idx.unsqueeze(1)] = 1.   # https://discuss.pytorch.org/t/fill-value-to-matrix-based-on-index/34698
        neg = (c * (1 - neg) + (1 - c) * neg).to(pos_samples.device)
        neg_samples.append(neg.unsqueeze(0))
    neg_samples = torch.cat(neg_samples, dim=0)
    
    return neg_samples, prop_val

def RMwGGIS_biased(pos_samples, neg_samples, energy_func):
    n, dim = pos_samples.shape
    pos = energy_func(pos_samples)
    neg = energy_func(neg_samples.reshape(-1, dim)).reshape(n, -1)
    loss = torch.mean(torch.logsumexp(2 * (pos - neg), dim=-1).exp())
    return loss

def RMwGGIS_unbiased(pos_samples, neg_samples, energy_func, prop_val):
    n, dim = pos_samples.shape
    pos = energy_func(pos_samples)
    neg = energy_func(neg_samples.reshape(-1, dim)).reshape(n, -1)

    # loss = torch.mean(dim * torch.mean((1/dim) * (2 * (pos - neg)).exp() / (prop_val + 1e-12), dim=-1))
    coef = 2 * (pos - neg) + ( np.log(1/dim + 1e-12) - (prop_val + 1e-12).log() )
    loss = torch.mean(dim * 1/coef.shape[-1] * torch.logsumexp(coef, dim=-1).exp())
    return loss

def main_loop(db, bm, inv_bm, energy_func):

    if cmd_args.phase == 'plot':
        print('loading energy_func from', cmd_args.energy_model_dump)
        energy_func.load_state_dict(torch.load(cmd_args.energy_model_dump, map_location=cmd_args.device))

        cmd_args.plot_size = 4.1
        plot_data(db, os.path.join(cmd_args.save_dir, 'data.pdf'))
        plot_heat(energy_func, bm, out_file=os.path.join(cmd_args.save_dir, '%s-heat.pdf' % cmd_args.data))
        plot_sampler(energy_func, inv_bm, os.path.join(cmd_args.save_dir, '%s-samples.pdf' % cmd_args.data))
        sys.exit()

    opt_energy = optim.Adam(energy_func.parameters(), lr=cmd_args.learning_rate * cmd_args.f_lr_scale)

    for epoch in range(cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.iter_per_epoch))
        for it in pbar:
            samples = float2bin(db.gen_batch(cmd_args.batch_size), bm)
            samples = torch.from_numpy(samples).to(cmd_args.device)

            prop_dit = get_proposal_dist(samples, energy_func)
            neg_samples, prop_val = importance_sampling(samples, prop_dit)

            loss = RMwGGIS_biased(samples, neg_samples, energy_func)
            # loss = RMwGGIS_unbiased(samples, neg_samples, energy_func, prop_val)
            
            loss.backward()
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(energy_func.parameters(), max_norm=cmd_args.grad_clip)
            opt_energy.step()
            if cmd_args.weight_clip > 0:
                for p in energy_func.parameters():
                    p.data.clamp_(-cmd_args.weight_clip, cmd_args.weight_clip)

            pbar.set_description('epoch: %d, loss: %.4f' % (epoch, loss.item()))
        
        if epoch and epoch % cmd_args.epoch_save == 0:
            torch.save(energy_func.state_dict(), os.path.join(cmd_args.save_dir, 'energy_func-%d.ckpt' % epoch))

            plot_data(db, os.path.join(cmd_args.save_dir, 'data.pdf'))
            plot_heat(energy_func, bm, out_file=os.path.join(cmd_args.save_dir, 'heat-%d.pdf' % epoch))
            plot_sampler(energy_func, inv_bm, os.path.join(cmd_args.save_dir, 'rand-samples-%d.pdf' % (epoch)))