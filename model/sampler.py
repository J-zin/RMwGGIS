import torch
import torch.nn as nn
import torch.nn.functional as F

def gibbs_step(orig_samples, axis, n_choices, score_func):
    orig_samples = orig_samples.clone()
    with torch.no_grad():
        cur_samples = orig_samples.clone().repeat(n_choices, 1)
        b = torch.LongTensor(list(range(n_choices))).to(cur_samples.device).view(-1, 1)
        b = b.repeat(1, orig_samples.shape[0]).view(-1)
        cur_samples[:, axis] = b
        score = score_func(cur_samples).view(n_choices, -1).transpose(0, 1)

        prob = F.softmax(-score, dim=-1)
        samples = torch.multinomial(prob, 1)
        orig_samples[:, axis] = samples.view(-1)
    return orig_samples

class GibbsSampler(nn.Module):
    def __init__(self, n_choices, discrete_dim, device):
        super(GibbsSampler, self).__init__()
        self.n_choices = n_choices
        self.discrete_dim = discrete_dim
        self.device = device

    def forward(self, score_func, num_rounds, num_samples=None, init_samples=None):
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = torch.randint(self.n_choices, (num_samples, self.discrete_dim)).to(self.device)

        with torch.no_grad():
            cur_samples = init_samples.clone()
            for r in range(num_rounds):
                for i in range(self.discrete_dim):
                    cur_samples = gibbs_step(cur_samples, i, self.n_choices, score_func)

        return cur_samples