from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .mpe.mpe_wrapper import MPEWrapper

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["mpe"] = partial(env_fn, env=MPEWrapper)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
