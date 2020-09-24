import torch.optim


class _Optimizer(object):

    def __init__(self, args, optimizer: torch.optim.Optimizer):
        super().__init__()
        if len(optimizer.param_groups) > 1:
            raise ValueError(f'Expected number of param_groups to be 1, got {len(optimizer.param_groups)} instead.')
        self.args = args
        self._optimizer = optimizer
        self._step = 0

    @classmethod
    def build(cls, args, params):
        return cls(args, params)

    @classmethod
    def add_args(cls, parser):
        pass

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def parameters(self):
        for group in self.param_groups:
            for param in group['params']:
                yield param

    def get_step(self):
        return self._step

    def state_dict(self):
        return {'_step': self._step, 'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['_step']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self._step += 1

    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
