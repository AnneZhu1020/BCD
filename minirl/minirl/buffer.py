import abc
from typing import Dict, Sequence
from threading import Lock

import numpy as np

from minirl.segment_tree import SumSegmentTree
from minirl.utils import get_scheduler
from minirl.type_utils import Schedulable


class Buffer:
    def __init__(self, max_size: int, sequence_length: int = 1):
        self._lock = Lock()
        self.max_size = max_size
        self.sequence_length = sequence_length
        self.max_num_sequences = max_size // sequence_length
        assert self.max_num_sequences > 0
        self.real_num_sequences = 0
        self.next_idx = 0
        self.storage = {}

    def __len__(self):
        return self.real_num_sequences

    def keys(self):
        return self.storage.keys()

    def update_next_idx(self, size):
        with self._lock:
            idx = self.next_idx
            self.next_idx = (self.next_idx + size) % self.max_num_sequences
            return idx

    def add(
        self, scheduler_step: int, data: Dict[str, np.ndarray], idx: int, size: int
    ):
        # Initialize if buffer is empty
        if not self.storage:
            for key, value in data.items():
                self.storage[key] = np.zeros(
                    shape=(
                        self.max_num_sequences,
                        *value.shape[1:],
                    ),
                    dtype=value.dtype,
                )
        # Add data
        idx_end = idx + size
        for key, value in data.items():
            if idx_end <= self.max_num_sequences:
                self.storage[key][idx:idx_end] = value
            else:
                first, second = np.split(value, (self.max_num_sequences - idx,), axis=0)
                self.storage[key][idx:] = first
                self.storage[key][: idx_end - self.max_num_sequences] = second
        # Set size
        with self._lock:
            self.real_num_sequences = min(
                self.real_num_sequences + size, self.max_num_sequences
            )

    def get_by_indices(self, indices: Sequence[int]):
        indices = np.asarray(indices, dtype=np.int64)
        return {key: value[indices] for key, value in self.storage.items()}

    def get_all(self):
        return self.storage


class ReplayBuffer(Buffer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample(self, scheduler_step: int, batch_size: int):
        raise NotImplementedError


class UniformReplayBuffer(ReplayBuffer):
    def sample(self, scheduler_step: int, batch_size: int):
        indices = np.random.randint(self.real_num_sequences, size=batch_size)
        batch = self.get_by_indices(indices)
        return batch


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        max_size: int,
        sequence_length: int = 1,
        alpha: Schedulable = 0.6,
        beta: Schedulable = 0.4,
        eps: float = 1e-6,
    ):
        super().__init__(max_size=max_size, sequence_length=sequence_length)
        self.alpha = get_scheduler(alpha)
        self.beta = get_scheduler(beta)
        self.eps = eps
        self.p_alpha = SumSegmentTree(size=self.max_num_sequences)
        self.max_priority = 1.0

    def add(
        self, scheduler_step: int, data: Dict[str, np.ndarray], idx: int, size: int
    ):
        super().add(data=data, idx=idx, size=size)
        idx_end = idx + size
        if idx_end <= self.max_num_sequences:
            indices = np.r_[idx:idx_end]
        else:
            indices = np.r_[
                idx : self.max_num_sequences, 0 : idx_end % self.max_num_sequences
            ]
        current_alpha = self.alpha.value(step=scheduler_step)
        self.p_alpha[indices] = self.max_priority ** current_alpha

    def sample_indices(self, batch_size: int):
        """
        Range [0, ptotal] is divided equally into k ranges,
        then a value is uniformly sampled from each range.
        """
        p_alpha_total = self.p_alpha.sum()
        p_alpha_range = np.linspace(0, p_alpha_total, num=batch_size, endpoint=False)
        shift = np.random.random_sample(size=batch_size) * p_alpha_total / batch_size
        mass = p_alpha_range + shift
        indices = self.p_alpha.find_prefixsum_idx(prefixsum=mass)
        # Clip to handle the case where mass[i] is very close to p_alpha_total
        # In that case, indices[i] will be self.p_alpha.capacity
        indices = np.clip(indices, None, self.real_num_sequences - 1)
        return indices

    def sample(self, scheduler_step: int, batch_size: int):
        indices = self.sample_indices(batch_size=batch_size)
        probs = self.p_alpha[indices] / self.p_alpha.sum()
        current_beta = self.beta.value(step=scheduler_step)
        weights = (probs * self.real_num_sequences) ** (-current_beta)
        weights /= np.max(weights)
        batch = self.get_by_indices(indices=indices)
        batch["weights"] = weights
        batch["indices"] = indices
        return batch

    def update_priorities(self, scheduler_step: int, indices, priorities):
        priorities += self.eps

        assert len(indices) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(indices) >= 0
        assert np.max(indices) < self.real_num_sequences

        current_alpha = self.alpha.value(step=scheduler_step)
        self.p_alpha[indices] = priorities ** current_alpha
        self.max_priority = max(self.max_priority, np.max(priorities))
