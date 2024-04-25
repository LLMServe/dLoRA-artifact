from abc import ABC, abstractmethod
import math
from typing import Tuple, Iterator
import random
import numpy as np

class ArrivalProcess(ABC):
    @abstractmethod
    def rate(self) -> float:
        """Return the mean arrival rate."""
        raise NotImplementedError()

    @abstractmethod
    def cv(self) -> float:
        """Return the coefficient of variation of the gap between
        the prompts."""
        raise NotImplementedError()

    @abstractmethod
    def get_iterator(self, start: float, duration: float,
                          seed: int = 0) -> Iterator[float]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"rate={self.rate()}, "
                f"cv={self.cv()})")

    def params(self) -> Tuple[str, str]:
        return self.rate(), self.cv()

class GammaProcess(ArrivalProcess):
    """Gamma arrival process."""
    def __init__(self, arrival_rate: float, cv: float):
        """Initialize a gamma arrival process.
        Args:
            arrival_rate: mean arrival rate.
            cv: coefficient of variation. When cv == 1, the arrival process is
                Poisson process.
        """
        self.rate_ = arrival_rate
        self.cv_ = cv
        self.shape = 1 / (cv * cv)
        self.scale = cv * cv / arrival_rate

    def rate(self) -> float:
        return self.rate_

    def cv(self) -> float:
        return self.cv_

    def get_iterator(self, start: float, duration: float, seed: int = 0) -> Iterator[float]:
        np.random.seed(seed)

        batch_size = max(int(self.rate_ * duration * 1.2), 1)
        intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
        pt = 0
        cur = start + intervals[0]
        end = start + duration
        while True:
            yield cur

            pt += 1
            if pt >= batch_size:
                intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
                pt = 0

            cur += intervals[pt]
    
class PoissonProcess(GammaProcess):
    """Poisson arrival process."""

    def __init__(self, arrival_rate: float):
        """Initialize a Poisson arrival process.
        Args:
            arrival_rate: The mean arrival rate.
        """
        super().__init__(arrival_rate, 1)


class Zipf:
    def __init__(self, min_val:int, max_val:int, theta: float) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.theta = theta
        self.zeta_n = 0
        self.n_for_zeta = 0
        self.zeta_2 = self.zeta(0, 2, theta, 0)
        self.alpha = 1.0 / (1.0 - theta)
        self.raise_zeta(max_val - min_val)
        self.eta = self.calc_eta()

    
    def zeta(self, last_num:int, cur_num:int, theta: float, last_zeta:float) -> float:
        zeta = last_zeta
        for i in range(last_num + 1, cur_num + 1):
            zeta += 1 / math.pow(i, theta)
        return zeta

    def raise_zeta(self, num: int) -> None:
        assert num >= self.n_for_zeta
        self.zeta_n = self.zeta(self.n_for_zeta, num, self.theta, self.zeta_n)
        self.n_for_zeta = num

    def calc_eta(self) -> float:
        return (1 - math.pow(2.0 / (self.max_val - self.min_val), 1 - self.theta)) / (1 - self.zeta_2 / self.zeta_n)

    def __iter__(self):
        return self

    def __next__(self):
        num = self.max_val - self.min_val
        assert num >= 2

        u = random.random()

        if num > self.n_for_zeta:
            self.raise_zeta(num)
            self.eta = self.eta()

        return int(self.min_val + num * math.pow(self.eta * u - self.eta + 1, self.alpha))

class Bimodal:
    def __init__(self, min_val:int, max_val:int, ratio: float) -> None:
        self._min_val = min_val
        self._max_val = max_val
        self._ratio = ratio
    
    def __iter__(self):
        return self

    def __next__(self):
        if random.random() < self._ratio:
            return self._min_val
        else:
            return self._max_val