from .utils.registry import Registry, build_from_cfg

ENVS = Registry("env")
AGENTS = Registry("agent")
NETS = Registry("net")
OPTIMIZERS = Registry("optimizer")


def build_env(cfg, default_args=None):
    return build_from_cfg(cfg, registry=ENVS, default_args=default_args)


def build_agent(cfg, default_args=None):
    return build_from_cfg(cfg, registry=AGENTS, default_args=default_args)


def build_net(cfg, default_args=None):
    return build_from_cfg(cfg, registry=NETS, default_args=default_args)


def build_optimizer(cfg, default_args=None):
    return build_from_cfg(cfg, registry=OPTIMIZERS, default_args=default_args)
