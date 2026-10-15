"""Exercise hand selection through the training dataset and real AI Worker FK."""

import draccus
import numpy as np
import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.state_free.ai_worker import AIWorkerConfig
from lerobot.state_free.relative_eef_pose import EEF_NAMES, RelativeEEFTransform
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.fixture(scope="module")
def joint_dataset(tmp_path_factory):
    config = AIWorkerConfig()
    names = [
        *config.left_arm_joint_names,
        *config.right_arm_joint_names,
        config.lift_joint_name,
        "gripper_l_joint1",
        "gripper_r_joint1",
    ]
    dataset = LeRobotDataset.create(
        "test/relative_hands",
        fps=10,
        root=tmp_path_factory.mktemp("relative_hands") / "dataset",
        features={
            key: {"dtype": "float32", "shape": (len(names),), "names": order}
            for key, order in ((OBS_STATE, names), (ACTION, names[::-1]))
        },
        use_videos=False,
    )
    for episode in range(2):
        for frame in range(5):
            joints = np.arange(len(names), dtype=np.float32) * 0.01 + (episode * 5 + frame) * 0.02
            dataset.add_frame({OBS_STATE: joints, ACTION: (joints + 0.05)[::-1].copy(), "task": "move"})
        dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset(dataset.repo_id, root=dataset.root)


@pytest.mark.parametrize(
    "hand,selection", [("both", slice(None)), ("left", slice(0, 10)), ("right", slice(10, 20))]
)
@pytest.mark.parametrize("eval_split", [0.0, 0.5])
def test_training_hand_selection(joint_dataset, hand, selection, eval_split):
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            f"--dataset.repo_id={joint_dataset.repo_id}",
            f"--dataset.root={joint_dataset.root}",
            f"--dataset.eval_split={eval_split}",
            "--policy.type=diffusion",
            f"--policy.hand={hand}",
            "--policy.device=cpu",
            "--policy.horizon=8",
            "--policy.n_action_steps=2",
        ],
    )
    train, evaluation = make_train_eval_datasets(cfg)
    both = LeRobotDataset(
        joint_dataset.repo_id,
        root=joint_dataset.root,
        episodes=train.episodes,
        delta_timestamps=train.delta_timestamps,
        state="relative",
    )
    size = 20 if hand == "both" else 10
    for dataset in (train, evaluation):
        if dataset is None:
            continue
        items = dataset.__getitems__(list(range(len(dataset))))
        for key in (OBS_STATE, ACTION):
            assert dataset.meta.features[key]["shape"] == (size,)
            assert dataset.meta.features[key]["names"] == EEF_NAMES[selection]
            assert all(item[key].shape[-1] == size for item in items)
            for stat in ("min", "max", "mean", "std"):
                np.testing.assert_allclose(
                    dataset.meta.stats[key][stat], both.meta.stats[key][stat][selection]
                )
        # The latest measured pose is the anchor, including at padded episode boundaries.
        for item in items:
            np.testing.assert_allclose(item[OBS_STATE][-1].reshape(-1, 10)[:, :3], 0, atol=1e-6)
    for index in range(len(train)):
        for key in (OBS_STATE, ACTION):
            torch.testing.assert_close(train[index][key], both[index][key][..., selection])
    assert (evaluation is None) == (eval_split == 0)
    # The public relative view must not overwrite the stored joint contract.
    raw = LeRobotDataset(joint_dataset.repo_id, root=joint_dataset.root)
    assert raw.meta.features == joint_dataset.meta.features
    torch.testing.assert_close(raw[0][ACTION], joint_dataset[0][ACTION])
    assert draccus.decode(DiffusionConfig, draccus.encode(cfg.policy)).hand == hand


@pytest.mark.parametrize("hand,selection", [("left", slice(0, 10)), ("right", slice(10, 20))])
def test_single_hand_accepts_only_selected_joints(joint_dataset, hand, selection):
    raw = joint_dataset[2]
    expected = RelativeEEFTransform(joint_dataset.features, joint_dataset.meta.stats)(raw)
    features, stats, item = {}, {}, {}
    for key in (OBS_STATE, ACTION):
        names = joint_dataset.features[key]["names"]
        indices = [i for i, name in enumerate(names) if name == "lift_joint" or f"_{hand[0]}_" in name]
        features[key] = {"shape": (len(indices),), "names": [names[i] for i in indices]}
        stats[key] = {stat: joint_dataset.meta.stats[key][stat][indices] for stat in ("min", "max")}
        item[key] = raw[key][indices]
    result = RelativeEEFTransform(features, stats, hand=hand)(item)
    for key in (OBS_STATE, ACTION):
        torch.testing.assert_close(result[key], expected[key][selection])


def test_hand_validation_and_legacy_default():
    assert DiffusionConfig().hand == "both"
    assert DiffusionConfig(state=None).hand == "both"
    with pytest.raises(ValueError, match="`hand` must be"):
        DiffusionConfig(hand="invalid")
    with pytest.raises(ValueError, match="requires state='relative'"):
        DiffusionConfig(state=None, hand="right")
