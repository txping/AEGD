import torch
from torch.optim import Optimizer

class AEGD(Optimizer):
    r"""Implements AEGD algorithm.
    It has been proposed in `AEGD: Adaptive Gradient Decent with Energy`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.1)
        c (float, optional): term added to the original objective function (default: 1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        aegdw (boolean, optional): whether to use the AEGDW variant of this algorithm
            (arxiv.org/abs/1711.05101) (default: False)

    .. _AEGD: Adaptive Gradient Decent with Energy:
    """

    def __init__(self, params, lr=0.1, c=1, weight_decay=0, aegdw=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, c=c, weight_decay=weight_decay, aegdw=aegdw)
        super(AEGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AEGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('aegdw', False)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        # Make sure the closure is defined and always called with grad enabled
        closure = torch.enable_grad()(closure)
        loss = closure()

        for group in self.param_groups:
            if not 0.0 < loss+group['c']:
                raise ValueError("c={} does not satisfy f(x)+c>0".format(group['c']))

            # Evaluate g(x)=(f(x)+c)^{1/2}
            sqrtloss = torch.sqrt(loss.detach() + group['c'])

            for p in group['params']:
                if p.grad is None:
                    continue
                df = p.grad
                if df.is_sparse:
                    raise RuntimeError('AEGD does not support sparse gradients')
                aegdw = group['aegdw']

                # Evaluate dg/dx = (df/dx) / (2*g(x))
                dg = df / (2 * sqrtloss)

                # Perform weight decay / L_2 regularization on g(x)
                if aegdw:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                else:
                    dg = dg.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['r'] = sqrtloss * torch.ones_like(p)

                r = state['r']

                r.div_(1 + 2 * group['lr'] * dg ** 2)
                p.addcmul_(r, dg, value=-2 * group['lr'])

        return loss
