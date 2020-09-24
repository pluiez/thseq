from thseq.optim.abs import _Optimizer
from thseq.optim.lr_schedulers.abs import _Scheduler
from . import register


@register('fixed')
class Fixed(_Scheduler):
    def __init__(self, args, lr, optimizer: _Optimizer):
        super().__init__(args, self.warmup_init_lr, optimizer)

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)

    def step(self, num_steps):
        """Update the learning rate before each optimizer step."""
        super(Fixed, self).step(num_steps)
        self.set_lr(self.lr)
        return self.lr
