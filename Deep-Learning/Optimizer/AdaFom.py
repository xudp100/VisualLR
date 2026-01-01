import math
import torch
from torch.optim.optimizer import Optimizer


class AdaFom(Optimizer):
    r"""Implements Adam algorithm with adaptive learning rate statistics.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaFom, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaFom, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            loss: The loss value (if closure is provided)
            stats (dict): Dictionary containing statistics about adaptive learning rates
                - 'adaptive_lr_max': Maximum value of 1/(sqrt(v_t) + eps)
                - 'adaptive_lr_min': Minimum value of 1/(sqrt(v_t) + eps)
                - 'adaptive_lr_mean': Mean value of 1/(sqrt(v_t) + eps)
        """
        loss = None
        if closure is not None:
            loss = closure()

        # 用于收集所有参数的自适应学习率
        all_adaptive_lrs = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)


                #adaptive_lr = 1.0 / (exp_avg_sq.sqrt().add_(group['eps']))
                adaptive_lr = 1.0 / ((exp_avg_sq.add_(group['eps'])).sqrt())


                all_adaptive_lrs.append(adaptive_lr.detach())

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                #exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # v̂_t = (1 - 1/t) v̂_{t-1} + (1/t) g_t²
                weight = 1.0 / state['step']
                exp_avg_sq.mul_(1 - weight).addcmul_(grad, grad, value=weight)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = ((exp_avg_sq.add_(group['eps']).sqrt()))

                else:
                    denom = ((exp_avg_sq.add_(group['eps']).sqrt()))  # v不使用偏差修正


                step_size = group['lr']
                p.data.addcdiv_(-step_size, exp_avg, denom)



        # 计算统计信息
        stats = {}
        if all_adaptive_lrs:

            all_adaptive_lrs_tensor = torch.cat([x.flatten() for x in all_adaptive_lrs])
            stats['adaptive_lr_max'] = all_adaptive_lrs_tensor.max().item()
            stats['adaptive_lr_min'] = all_adaptive_lrs_tensor.min().item()

        return loss, stats