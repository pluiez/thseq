from thseq.optim.abs import _Optimizer
from thseq.optim.lr_schedulers.abs import _Scheduler
from . import register


def f(step, init_lr, warmup_steps, climb_factor, decay_factor):
    if warmup_steps <= 1:
        return decay_factor * (step ** (-0.5))
    else:
        if step <= warmup_steps:
            lr = init_lr + climb_factor * (step - 1)
        else:
            lr = decay_factor * (step ** (-0.5))
        return lr


@register('inverse_sqrt')
class InverseSquaredRoot(_Scheduler):
    """
    Linearly warming-up and sqrt decreasing.

    lr=7e-4 and lr=5e-4 is almost equivalent to the maximum lr empirically used in noam scheduler
    for base and big model.
    """

    def __init__(self, args, lr, optimizer: _Optimizer):
        if args.warmup_steps < 1:
            raise ValueError(f'warmup-steps ({args.warmup_steps}) must be at least 1.')
        if args.warmup_init_lr > lr:
            raise ValueError(f'expected warmup-init-lr ({args.warmup_init_lr}) < lr ({lr}).')
        self.warmup_init_lr = args.warmup_init_lr
        self.warmup_end_lr = lr
        self.warmup_steps = args.warmup_steps
        if self.warmup_steps > 1:
            if args.warmup_init_lr > 0:
                self.climb_factor = (lr - args.warmup_init_lr) / (args.warmup_steps - 1)
            else:
                self.climb_factor = lr / args.warmup_steps
                self.warmup_init_lr = self.climb_factor
        else:
            self.warmup_init_lr = 0
            self.climb_factor = 0
        self.decay_factor = lr * (self.warmup_steps ** 0.5)

        super().__init__(args, self.warmup_init_lr, optimizer)

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--warmup-init-lr', type=float, default=1e-7)
        parser.add_argument('--warmup-steps', type=int, default=4000)

    def step(self, num_steps):
        """Update the learning rate before each optimizer step."""
        super(InverseSquaredRoot, self).step(num_steps)
        lr = f(num_steps, self.warmup_init_lr, self.warmup_steps, self.climb_factor, self.decay_factor)
        self.set_lr(lr)
        return lr
