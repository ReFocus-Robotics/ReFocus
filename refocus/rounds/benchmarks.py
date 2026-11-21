"""Benchmark settings shared by the round coordinator, the Scheduler, and the recorders."""

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from refocus import libero, robocasa
from refocus.rounds import libero_codec, robocasa_codec


@dataclass(frozen=True)
class Benchmark:
    name: str
    title: str
    tasks_module: ModuleType
    codec: ModuleType
    default_config: str
    experiment_name: str
    chunk: int
    horizon: int
    reward_variant_env: str

    @property
    def task_stages(self):
        return self.tasks_module.TASK_STAGES

    @property
    def milestone_checks(self):
        return self.tasks_module.MILESTONE_CHECKS

    def task_evidence(self, task):
        return self.tasks_module.task_evidence(task)

    @property
    def cameras(self):
        return " and ".join(self.codec.CAMERA_NAMES)

    def tasks(self, config_dir):
        if self.name == "robocasa365":
            import yaml
            path = Path(config_dir) / "env" / "robocasa365_composite_seen.yaml"
            return list(yaml.safe_load(path.read_text())["task_names"])
        return list(libero.TASKS)

    def task_overrides(self, tasks):
        """Hydra overrides restricting training and evaluation to ``tasks``."""
        if self.name == "robocasa365":
            names = "[" + ",".join(tasks) + "]"
            return [f"env.train.task_names={names}", f"env.eval.task_names={names}"]
        ids = "[" + ",".join(str(libero.TASKS.index(task)) for task in tasks) + "]"
        return [f"env.train.task_id_filter={ids}", f"env.eval.task_id_filter={ids}"]


BENCHMARKS = {
    "robocasa365": Benchmark(
        name="robocasa365",
        title="RoboCasa365 Composite-Seen",
        tasks_module=robocasa,
        codec=robocasa_codec,
        default_config="robocasa365_ppo_openpi_pi05_refocus_16gpu",
        experiment_name="robocasa_rounds",
        chunk=50,
        horizon=2900,
        reward_variant_env="STRUCTRL_REWARD_VARIANT",
    ),
    "libero_10": Benchmark(
        name="libero_10",
        title="LIBERO-Long",
        tasks_module=libero,
        codec=libero_codec,
        default_config="libero_10_ppo_openpi_pi05_refocus",
        experiment_name="libero_rounds",
        chunk=10,
        horizon=1024,
        reward_variant_env="LIBERO_REWARD_VARIANT",
    ),
}
