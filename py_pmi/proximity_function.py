import numpy as np
from scipy.spatial.distance import pdist, squareform


class ProximityFunction:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def proximity_function(self, r):
        raise NotImplementedError(
            "proximity_function method should be implemented in subclass."
        )

    def get_parameters(self):
        return [v for v in self.__dict__.values()]

    def compute_proximity_matrix(self, distances):
        s = distances.shape
        if len(s) == 1:
            return squareform(self.proximity_function(distances))
        else:
            n_total = s[0]
            n_atoms = int((1 + np.sqrt(s[1] * 8 + 1)) / 2)
            proximities = self.proximity_function(distances)
            m = np.swapaxes(
                [
                    np.concatenate(
                        (
                            np.zeros((n_total, i + 1)),
                            proximities[
                                :,
                                i * n_atoms
                                - i * (i + 1) // 2 : (i + 1) * n_atoms
                                - (i + 1) * (i + 2) // 2,
                            ],
                        ),
                        axis=1,
                    )
                    for i in range(n_atoms)
                ],
                0,
                1,
            )
            m = m + np.transpose(m, axes=(0, 2, 1))
            return m

    def compute_proximity_matrix_scipy(self, distances):
        s = distances.shape
        n_atoms = int((1 + np.sqrt(s[1] * 8 + 1)) / 2)
        m = np.ndarray(shape=(s[0], n_atoms, n_atoms))
        for i, v in enumerate(distances):
            proximities = self.proximity_function(v)
            m[i, :, :] = squareform(proximities)

        return m


class ProximityFunction0(ProximityFunction):
    def __init__(self, r0, n) -> None:
        super().__init__(r0=r0, n=n)
        self.function_id = 0

    def proximity_function(self, r):
        return np.where(
            r > self.r0,
            0.0,
            (self.r0 / (self.r0 + r) - r / (2 * self.r0)) ** self.n,
        )

    def get_latex_parameters(self):
        return f"$r_0 = {self.r0}, n = {self.n}$"

    def get_latex_function(self):
        return (
            r"$\left(\frac{"
            + f"{self.r0}"
            + r"}{"
            + f"{self.r0}"
            + r"+r}-\frac{r}{2 \times"
            + f"{self.r0}"
            + r"}\right)^"
            + f"{self.n}$"
        )

    def get_latex_function_shape(self):
        return r"$\left(\frac{r_0}{r_0+r}-\frac{r}{2r_0}\right)^n$"


class ProximityFunction1(ProximityFunction):
    def __init__(self, r0, n) -> None:
        super().__init__(r0=r0, n=n)
        self.function_id = 1

    def proximity_function(self, r):
        return np.where(r >= self.r0, 0.0, (1.0 - r / self.r0) ** self.n)

    def get_latex_parameters(self):
        return f"$r_0 = {self.r0}, n = {self.n}$"

    def get_latex_function(self):
        return r"$\left(1-\frac{r}{" + f"{self.r0}" + r"}\right)^" + f"{self.n}$"

    def get_latex_function_shape(self):
        return r"$\left(1-\frac{r}{r_0}\right)^n$"
