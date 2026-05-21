available_indicators = []
_indicator_impls = {}


def indicator(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def indicator_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_copy(bn=bn).to(device)
            else:
                net = net_orig.to(device)
            if name == 'NASWOT' or name == 'SC':
                ret = func(net,  *args, **kwargs, **impl_args)
            else:
                ret = func(net, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc
                import torch
                del net
                gc.collect()
                torch.cuda.empty_cache()
            return ret

        global _indicator_impls
        if name in _indicator_impls:
            raise KeyError(f'Duplicated indicator! {name}')
        available_indicators.append(name)
        _indicator_impls[name] = indicator_impl
        return func
    return make_impl


def calc_indicator(name, net, device, *args, **kwargs):
    return _indicator_impls[name](net, device, *args, **kwargs)


def load_all():
    from . import SC
    # from . import NASWOT
    # from . import dss
    # from . import fisher
load_all()
