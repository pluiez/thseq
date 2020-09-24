from thseq.optim import _Optimizer


class _Scheduler(object):
    """
    Note: Contrary to PyTorch's LRScheduler,
        this scheduler's step/step_epoch should be invoked before each step/epoch.
    """

    def __init__(self, args, lr, optimizer: _Optimizer):
        super().__init__()
        self.args = args
        # initial learning rate
        self.base_lr = lr
        self.optimizer = optimizer
        self.best = None

    @classmethod
    def build(cls, args, lr, optimizer):
        return cls(args, lr, optimizer)

    @classmethod
    def add_args(cls, parser):
        pass

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {'best': self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict['best']

    def set_lr(self, lr):
        self.optimizer.set_lr(lr)

    def get_lr(self):
        return self.optimizer.get_lr()

    def step_epoch(self, epoch, val_score=None):
        """Update the learning rate *before* the given epoch."""
        if val_score is not None:
            if self.best is None:
                self.best = val_score
            else:
                self.best = max(self.best, val_score)
        return self.get_lr()

    def step(self, num_steps):
        """Update the learning rate *before* each update."""
        assert num_steps >= 1
        return self.get_lr()
