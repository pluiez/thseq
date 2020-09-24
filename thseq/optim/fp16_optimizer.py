import torch
import torch.nn as nn

import thseq.utils as utils
from .abs import _Optimizer

__all__ = ['GradScaler', 'FP16Optimizer']


def _has_overflow(optimizer):
    for group in optimizer.param_groups:
        if utils.has_overflow(group["params"]):
            return True
    return False


class GradScaler(object):
    def __init__(self,
                 init_scale=2. ** 7,
                 growth_factor=2.0,
                 backoff_factor=0.5,
                 growth_interval=2000,
                 min_loss_scale=2 ** (-5),
                 enabled=True) -> None:
        self._enabled = enabled
        if self._enabled:
            assert growth_factor > 1.0, "The growth factor must be > 1.0."
            assert backoff_factor < 1.0, "The backoff factor must be < 1.0."

        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._min_loss_scale = min_loss_scale
        self._growth_tracker = 0
        self._optim_states = {}

    def scale(self, tensors):
        if not self._enabled:
            return tensors

        scale = self.get_scale()
        if torch.is_tensor(tensors):
            return scale * tensors
        else:
            return [scale * tensor for tensor in tensors]

    def _unscale_grads(self, optimizer):
        if not self._enabled:
            return

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.mul_(1.0 / self._scale)

    def unscale_(self, optimizer):
        if not self._enabled:
            return

        state = self._get_optimizer_state(optimizer)
        if state['unscaled']:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        self._unscale_grads(optimizer)
        state['unscaled'] = True

    def step(self, optimizer):
        if not self._enabled:
            optimizer.step()
            return

        overflow = _has_overflow(optimizer)
        if not overflow:
            if not self._get_optimizer_state(optimizer)['unscaled']:
                self.unscale_(optimizer)
            optimizer.step()
        self._update_scale(overflow)
        if overflow:
            raise OverflowError

    def _update_scale(self, overflow):
        if overflow:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker > self._growth_interval:
                self._growth_tracker = 0
                self._scale *= self._growth_factor

    def get_scale(self):
        return self._scale

    def _get_optimizer_state(self, optimizer):
        return self._optim_states.get(id(optimizer), {'unscaled': False, 'overflow': False})

    def state_dict(self):
        return {'scale': self._scale, 'growth_tracker': self._growth_tracker}

    def load_state_dict(self, state):
        self._scale = state['scale']
        self._growth_tracker = state['growth_tracker']


def get_fp32_params(fp16_params, flatten):
    fp16_params = list(filter(lambda p: p.requires_grad, fp16_params))
    if flatten:
        total_size = sum(p.numel() for p in fp16_params)
        fp32_params = torch.zeros(total_size, dtype=torch.float, device=fp16_params[0].device)
        offset = 0
        for p16 in fp16_params:
            numel = p16.numel()
            fp32_params.data[offset:offset + numel].copy_(p16.data.view(-1))
            offset += numel
        fp32_params = nn.Parameter(fp32_params)
        fp32_params.grad = fp32_params.data.new_zeros(total_size)
    else:
        fp32_params = []
        for p16 in fp16_params:
            fp32 = nn.Parameter(p16.data.float())
            fp32.grad = fp32.data.new_zeros(fp32.shape)
            fp32_params.append(fp32)
    return fp16_params, fp32_params


class FP16Optimizer(_Optimizer):
    def __init__(self, args, fp16_params, fp32_optimizer_builder):
        flatten = bool(args.fp16_flatten)
        fp16_params, fp32_params = get_fp32_params(fp16_params, flatten)
        super().__init__(args, fp32_optimizer_builder([fp32_params] if flatten else fp32_params))
        self._flatten = flatten
        self.fp16_params = fp16_params
        self.fp32_params = fp32_params

    def fp16_grads_to_fp32_grads(self):
        if self._flatten:
            offset = 0
            for p16 in self.fp16_params:
                numel = p16.numel()
                if p16.grad is None:
                    self.fp32_params.grad.data[offset:offset + numel].zero_()
                else:
                    self.fp32_params.grad.data[offset:offset + numel].copy_(p16.grad.data.view(-1))
                offset += numel
        else:
            for p16, p32 in zip(self.fp16_params, self.fp32_params):
                if p16.grad is None:
                    p32.grad.data.zero_()
                else:
                    p32.grad.data.copy_(p16.grad.data)

    def fp32_params_to_fp16_params(self):
        if self._flatten:
            offset = 0
            for p16 in self.fp16_params:
                numel = p16.numel()
                p16.data.copy_(self.fp32_params.data[offset:offset + numel].view_as(p16.data))
                offset += numel
        else:
            for p16, p32 in zip(self.fp16_params, self.fp32_params):
                p32.data.copy_(p16.data)

    def zero_grad(self):
        for p in self.fp16_params:
            p.grad = None
        if self._flatten:
            self.fp32_params.grad.zero_()
        else:
            for p in self.fp32_params:
                p.grad.zero_()

    def step(self):
        super().step()
        self.fp32_params_to_fp16_params()

    def state_dict(self):
        state = super().state_dict()
        if self._flatten:
            state['fp32_params'] = self.fp32_params.data
        else:
            state['fp32_params'] = [p.data for p in self.fp32_params]
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self._flatten:
            self.fp32_params.data.copy_(state_dict['fp32_params'])
        else:
            for p, p_ in zip(self.fp32_params, state_dict['fp32_params']):
                p.data.copy_(p_)
