REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_runner_div_time import ParallelRunnerDivT
REGISTRY["parallel_div_time"] = ParallelRunnerDivT
