# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            # self本身就是一个字典，作为参数传递出去
            _register_generic(self, module_name, module)
            return

        # used as decorator
        '''
        此装饰器作用：返回的还是原函数，不过再dict注册了一下
        等价于eg: 
        build_resnet_backbone = registry.BACKBONES.register("R-50-C4")(build_resnet_backbone)
        build_resnet_backbone = register_fn(build_resnet_backbone)
        所以会执行里面的语句
        '''
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            # 此处如果不返回fn，还是执行了注册的，不过后面被注册的函数就没办法调用了
            return fn

        return register_fn
