__all__ = ['is_inference', 'set_inference', 'inference_mode']

import contextlib

_is_inference = False


def is_inference():
    return _is_inference


def set_inference(mode: bool):
    global _is_inference
    _is_inference = mode


@contextlib.contextmanager
def inference_mode(mode: bool = True):
    state = is_inference()
    set_inference(mode)
    try:
        yield
    finally:
        set_inference(state)
