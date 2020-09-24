from thseq.optim.abs import _Optimizer
from thseq.optim.lr_schedulers.abs import _Scheduler


# @register('exp')
class Exponential(_Scheduler):

    def __init__(self, args, lr, optimizer: _Optimizer, groups=None):
        self.gamma = args.gamma
        super().__init__(args, lr, optimizer, groups)

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--gamma', type=float, default=0.5)

    def step_epoch(self, epoch, val_score=None):
        super().step_epoch(epoch, val_score)
        lr = self.base_lr * self.gamma ** epoch
        self.set_lr(lr)
        return lr
