import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(self, params,lr = 1e-3,betas= (0.9,0.999),eps=1e-8,weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW,self).__init__(params, defaults)

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # parameters of the group
            for p in group['params']:
                if p.grad is None:
                    continue

                # get gradient
                grad = p.grad

                # state
                state = self.state[p]

                # instialize state
                if len(state) == 0:
                    state['step'] = 0
                    # first moment
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # second moment
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state['m'], state['v']

                state['step'] += 1

                t = state['step']

                # update moments
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                b_corr1 = 1 - beta1 ** t
                b_corr2 = 1 - beta2 ** t

                alpha_t = lr * math.sqrt(b_corr2) / b_corr1

                # update parameters
                p.addcdiv_(m, v.sqrt().add(eps), value=-alpha_t)

                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)
            
        return loss


def get_lr_cosine_schedule(t,alpha_max,alpha_min,T_w,T_c):
    # warmup
    if t < T_w:
        alpha_t = alpha_max * t / T_w
    # cosine annealing
    elif T_w <= t < T_c:

        #annealing progress
        progress = (t - T_w) / (T_c - T_w)
        cosine_val = 1 + math.cos(math.pi * progress)
        alpha_t = alpha_min + 0.5 * cosine_val * (alpha_max - alpha_min)

    # post annealing
    else:
        alpha_t = alpha_min
    
    return alpha_t

def gradient_clipping(parameters, max_norm,eps=1e-6):
    # remove parameters with no gradients
    params_grad = [p for p in parameters if p.grad is not None]

    if len(params_grad) == 0:
        return
    
    # l2 for all gradients
    #total_norm_sq =0.0

    total_norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in params_grad)
    )

    clip_factor = max_norm / (total_norm + eps)

    if clip_factor < 1.0:
        for p in params_grad:
            p.grad.data.mul_(clip_factor)
    
    return float(total_norm)