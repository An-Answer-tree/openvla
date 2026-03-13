"""PyTorch dataset for single-view LIBERO HDF5 finetuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Type

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from libero.libero import benchmark as libero_benchmark
from libero_finetune.constants import ACTION_NORMALIZATION_MASK, NORMALIZATION_EPS
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import IGNORE_INDEX
from prismatic.vla.action_tokenizer import ActionTokenizer


@dataclass(frozen=True)
class TransitionRecord:
    """Index record for a single transition inside a LIBERO HDF5 file."""

    file_path: Path
    demo_key: str
    step_index: int
    language: str


def _demo_sort_key(demo_key: str) -> int:
    """Sorts demo keys like `demo_0`, `demo_1`, ... numerically."""
    return int(demo_key.split("_")[-1])


def _convert_actions(actions: np.ndarray) -> np.ndarray:
    """Converts raw LIBERO gripper commands to OpenVLA training convention."""
    converted = np.asarray(actions, dtype=np.float32).copy()
    converted[:, -1] = 1.0 - np.clip(converted[:, -1], 0.0, 1.0)
    return converted


def _compute_action_statistics(actions: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes dataset statistics matching OpenVLA's expected schema."""
    return {
        "mean": np.mean(actions, axis=0).astype(np.float32),
        "std": np.std(actions, axis=0).astype(np.float32),
        "min": np.min(actions, axis=0).astype(np.float32),
        "max": np.max(actions, axis=0).astype(np.float32),
        "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32),
        "mask": np.asarray(ACTION_NORMALIZATION_MASK, dtype=bool),
    }


class LiberoMultiviewHDF5Dataset(Dataset):
    """Loads single-view transition samples from the multiview LIBERO HDF5 dataset."""

    def __init__(
        self,
        data_root_dir: Path,
        benchmark_name: str,
        camera_view: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        rotate_image_180: bool = True,
    ) -> None:
        self.data_root_dir = Path(data_root_dir)
        self.benchmark_name = benchmark_name
        self.camera_view = camera_view
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.rotate_image_180 = rotate_image_180

        self._file_cache: Dict[Path, h5py.File] = {}
        self.records, self.dataset_statistics = self._build_index_and_statistics()
        self._action_stats = {
            key: np.asarray(value).copy()
            for key, value in self.dataset_statistics[self.benchmark_name]["action"].items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        hdf5_file = self._get_hdf5_file(record.file_path)

        demo_group = hdf5_file["data"][record.demo_key]
        image = np.asarray(demo_group["obs"][self.camera_view][record.step_index], dtype=np.uint8)
        action = np.asarray(demo_group["actions"][record.step_index], dtype=np.float32)

        image = self._prepare_image(image)
        action = self._normalize_action(_convert_actions(action[None, ...])[0])

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {record.language.lower()}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        pixel_values = self.image_transform(Image.fromarray(image))

        labels_tensor[: -(len(action) + 1)] = IGNORE_INDEX

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "dataset_name": self.benchmark_name,
        }

    def __getstate__(self) -> Dict:
        """Drops open file handles before the dataset is copied into DataLoader workers."""
        state = self.__dict__.copy()
        state["_file_cache"] = {}
        return state

    def close(self) -> None:
        """Closes any cached HDF5 files."""
        for hdf5_file in self._file_cache.values():
            try:
                hdf5_file.close()
            except OSError:
                continue
        self._file_cache.clear()

    def _build_index_and_statistics(self) -> tuple[List[TransitionRecord], Dict[str, Dict]]:
        """Builds the transition index and dataset statistics in one pass."""
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.benchmark_name]()

        records: List[TransitionRecord] = []
        action_chunks: List[np.ndarray] = []
        num_trajectories = 0
        num_transitions = 0

        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            file_path = self.data_root_dir / self.benchmark_name / f"{task.name}_demo.hdf5"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing LIBERO dataset file: {file_path}")

            with h5py.File(file_path, "r") as hdf5_file:
                demo_group = hdf5_file["data"]
                for demo_key in sorted(demo_group.keys(), key=_demo_sort_key):
                    actions = _convert_actions(np.asarray(demo_group[demo_key]["actions"], dtype=np.float32))
                    if self.camera_view not in demo_group[demo_key]["obs"]:
                        raise KeyError(
                            f"Camera view `{self.camera_view}` not found in `{file_path}` / `{demo_key}`."
                        )

                    action_chunks.append(actions)
                    num_trajectories += 1
                    num_transitions += actions.shape[0]

                    for step_index in range(actions.shape[0]):
                        records.append(
                            TransitionRecord(
                                file_path=file_path,
                                demo_key=demo_key,
                                step_index=step_index,
                                language=task.language,
                            )
                        )

        all_actions = np.concatenate(action_chunks, axis=0)
        dataset_statistics = {
            self.benchmark_name: {
                "action": _compute_action_statistics(all_actions),
                "num_trajectories": num_trajectories,
                "num_transitions": num_transitions,
            }
        }

        return records, dataset_statistics

    def _get_hdf5_file(self, file_path: Path) -> h5py.File:
        """Returns a cached HDF5 file handle."""
        if file_path not in self._file_cache:
            self._file_cache[file_path] = h5py.File(file_path, "r")
        return self._file_cache[file_path]

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Applies LIBERO image preprocessing before the OpenVLA transform."""
        if self.rotate_image_180:
            image = image[::-1, ::-1]
        return np.ascontiguousarray(image)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalizes continuous actions to the OpenVLA training range."""
        normalized = action.copy()
        mask = self._action_stats["mask"]
        low = self._action_stats["q01"]
        high = self._action_stats["q99"]

        normalized[mask] = np.clip(
            2.0 * (normalized[mask] - low[mask]) / (high[mask] - low[mask] + NORMALIZATION_EPS) - 1.0,
            -1.0,
            1.0,
        )

        zero_mask = self._action_stats["min"] == self._action_stats["max"]
        normalized[zero_mask] = 0.0
        return normalized.astype(np.float32)
