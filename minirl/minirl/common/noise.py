from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class ActionNoise(ABC):
    def __init__(self) -> None:
        super(ActionNoise, self).__init__()

    def reset(self, flags: np.ndarray, size: Tuple[int]) -> None:
        """
        call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass


class GaussianActionNoise(ActionNoise):
    """
    Gaussian action noise
    """

    def __init__(self, mu: float = 0.0, sigma: float = 0.1) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self, size: Tuple[int]) -> np.ndarray:
        return np.random.normal(loc=self.mu, scale=self.sigma, size=size)


class OUActionNoise(ActionNoise):
    """
    Ornstein Uhlenbeck action noise
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.2,
        theta: float = 0.15,
        dt: float = 1e-2,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.alpha = theta * dt
        self.beta = sigma * np.sqrt(dt)
        self.prev_noise = None

    def reset(self, flags: np.ndarray, size: Tuple[int]) -> None:
        if self.prev_noise is None:
            self.prev_noise = np.full(size, self.mu)
        else:
            self.prev_noise[flags] = self.mu

    def __call__(self, size: Tuple[int]) -> np.ndarray:
        self.prev_noise += self.alpha * (self.mu - self.prev_noise)
        self.prev_noise += self.beta * np.random.normal(size=size)
        return self.prev_noise
