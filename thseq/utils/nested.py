import collections.abc

__all__ = ['map']


def map(func, x, predicate, default=None, default_is_none=False):
    """
    Apply `func` to every element of x following its structure when `predicate` is evaluated True on `x`.
    For elements evaluated False, `default` and `default_none` help decide what is returned to fulfill the structure.
      1. Item remains unchanged if `default` is None and `default_none` is False.
      2. Item is set to `default` if default is not None. In this case, `default_none` is redundant.
      3. Item is set to None if `default` is None and `default_none` is True.
    Args:
        func: callable function to apply to an element.
        x: nested structure.
        predicate: callable
        default: default value for element evaluated False
        default_is_none: bool

    Returns:

    """

    def rec(x):
        if predicate(x):
            y = func(x)
        elif isinstance(x, collections.abc.Mapping):
            y = x.__class__((k, rec(v)) for k, v in x.items())
        elif isinstance(x, tuple) and hasattr(x, '_fields'):  # namedtuple
            y = type(x)(*(rec(_) for _ in x))
        elif isinstance(x, collections.abc.Sequence) and not isinstance(x, str):
            y = x.__class__([rec(_) for _ in x])
        else:
            y = x if default is None and not default_is_none else default
        return y

    return rec(x)
