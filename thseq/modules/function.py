def maybe_norm(normalizer, x, is_pre: bool, enable_post: bool):
    if not enable_post:
        return normalizer(x) if is_pre else x
    else:
        return normalizer(x) if not is_pre else x
