import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples/robocasa365/rlinf/retain_ppo_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("retain_ppo_checkpoints", MODULE_PATH)
RETENTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETENTION)


def make_checkpoint(root: Path, step: int, *, complete: bool = True) -> Path:
    checkpoint = root / f"global_step_{step}" / "actor"
    (checkpoint / "dcp_checkpoint").mkdir(parents=True)
    (checkpoint / "dcp_checkpoint/.metadata").write_bytes(b"metadata")
    (checkpoint / "dcp_checkpoint/__0_0.distcp").write_bytes(b"shard")
    if complete:
        (checkpoint / "model_state_dict").mkdir()
        (checkpoint / "model_state_dict/full_weights.pt").write_bytes(b"weights")
    return checkpoint.parent


def append_metrics(path: Path, step: int, success: float, episodes: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for tag, value in (("eval/success_once", success), ("eval/num_trajectories", episodes)):
            output.write(json.dumps({"step": step, "tag": tag, "value": value}) + "\n")


class CheckpointRetentionTests(unittest.TestCase):
    def test_keeps_best_and_latest_and_prunes_older_complete_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoints"
            metrics = root / "metrics/per_step_metrics.jsonl"
            for step, success in ((5, 0.6), (10, 0.8), (15, 0.7)):
                make_checkpoint(root, step)
                append_metrics(metrics, step, success)

            stable = {}
            self.assertIsNone(RETENTION.retain_checkpoints(root, metrics, stable_observations=stable))
            result = RETENTION.retain_checkpoints(root, metrics, stable_observations=stable)

            self.assertEqual((result["best_step"], result["last_step"]), (10, 15))
            self.assertEqual(os_readlink(root / "best"), "global_step_10")
            self.assertEqual(os_readlink(root / "last"), "global_step_15")
            self.assertTrue((root / "global_step_10").is_dir())
            self.assertTrue((root / "global_step_15").is_dir())
            self.assertFalse((root / "global_step_5").exists())

    def test_does_not_prune_when_eval_count_or_checkpoint_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoints"
            metrics = root / "metrics/per_step_metrics.jsonl"
            make_checkpoint(root, 5)
            append_metrics(metrics, 5, 0.7)
            incomplete = make_checkpoint(root, 10, complete=False)
            append_metrics(metrics, 10, 0.9)
            stable = {}

            for _ in range(3):
                result = RETENTION.retain_checkpoints(root, metrics, stable_observations=stable)

            self.assertEqual((result["best_step"], result["last_step"]), (5, 5))
            self.assertTrue((root / "global_step_5").exists())
            self.assertTrue((root / "global_step_10").exists())
            self.assertEqual(os_readlink(root / "best"), "global_step_5")
            self.assertEqual(RETENTION.checkpoint_signature(incomplete), None)


def os_readlink(path: Path) -> str:
    return path.readlink().as_posix()


if __name__ == "__main__":
    unittest.main()
