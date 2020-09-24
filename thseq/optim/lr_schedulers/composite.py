import itertools


class Composite(object):
    def __init__(self, lr_schedulers):
        super().__init__()
        self.lr_schedulers = list(lr_schedulers)

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {'lr_schedulers': [x.state_dict() for x in self.lr_schedulers]}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        for i, state in enumerate(state_dict['lr_schedulers']):
            self.lr_schedulers[i].load_state_dict(state)

    def get_lr(self):
        return list(itertools.chain(x.get_lr() for x in self.lr_schedulers))

    def get_step(self):
        return self.lr_schedulers[0].get_step()

    def step_epoch(self, epoch, val_scores=None):
        """Update the learning rate *before* the given epoch."""
        val_scores = val_scores or [None] * len(self.lr_schedulers)
        for x, score in itertools.zip_longest(self.lr_schedulers, val_scores):
            x.step_epoch(epoch, score)
        return self.get_lr()

    def step(self):
        """Update the learning rate *before* each update."""
        for x in self.lr_schedulers:
            x.step()
        return self.get_lr()
