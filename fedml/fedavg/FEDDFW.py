import torch
import torch.optim as optim
from collections import defaultdict
from torch.optim.optimizer import required
class DFW(optim.Optimizer):
    r"""
    Implements Deep Frank Wolfe: https://arxiv.org/abs/1811.07591.
    Nesterov momentum is the *standard formula*, and differs
    from pytorch NAG implementation.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eta (float): initial learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): small constant for numerical stability (default: 1e-5)

    Example:
        >>> optimizer = DFW(model.parameters(), eta=1, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.zero_grad()
        >>> loss_value = loss_fn(model(input), target)
        >>> loss_value.backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, eps=1e-5):
        if lr is not required and lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(DFW, self).__init__(params, defaults)
        self.eps = eps

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

    @torch.no_grad()
    def step(self, loss):
        # Ensure loss is a tensor
        if not isinstance(loss, torch.Tensor):
            raise ValueError("Loss should be a tensor")
        # Compute gradients
        loss.backward()

        w_dict = defaultdict(dict)
        for group in self.param_groups:
            wd = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                w_dict[param]['delta_t'] = param.grad.data
                w_dict[param]['r_t'] = wd * param.data

        self._line_search(loss.item(), w_dict)

        for group in self.param_groups:
            eta = group['lr']
            mu = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                param.data -= eta * (r_t + self.gamma * delta_t)

                if mu:
                    z_t = state['momentum_buffer']
                    z_t *= mu
                    z_t -= eta * self.gamma * (delta_t + r_t)
                    param.data += mu * z_t
        return loss.item()

    @torch.no_grad()
    def _line_search(self, loss, w_dict):
        """
        Computes the line search in closed form.
        """

        num = torch.tensor(loss, device=self.param_groups[0]['params'][0].device)
        denom = torch.tensor(0.0, device=self.param_groups[0]['params'][0].device)

        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                num -= lr * torch.sum(delta_t * r_t)
                denom += lr * delta_t.norm() ** 2

        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))
