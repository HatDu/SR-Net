def get_mask_func(name):
    print(name)
    if name == 'cartesian_1d':
        from dataset.mask.cartesian_1d import MaskFunc
    elif name == 'poisson_2d':
        from dataset.mask.poisson_2d import MaskFunc
    else:
        raise 'no such kind of mask function'
    return MaskFunc                     