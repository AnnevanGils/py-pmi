from py_pmi.proximity_function import ProximityFunction
import os


def get_savefiles(
    savedir: str, r0: float, n: float, function_id: int, order: int, dataset_str_id: str
) -> dict:
    return {
        strict_order: savedir
        + "/"
        + f"{dataset_str_id}"
        + f"/r0_{r0}_n_{n}"
        + f"/prox_func_{function_id}"
        + f"/inv_f_{function_id}_r0_{r0}_n_{n}_order_{strict_order}.npy"
        for strict_order in range(1, order + 1)
    }


class InvariantSet:
    def __init__(
        self, dataset_str_id: str, order: int, proximity_function: ProximityFunction
    ) -> None:
        self.order = order
        self.proximity_function = proximity_function

        if dataset_str_id.lower() == "co2":
            self.dataset_str_id = dataset_str_id.lower()
            self.nr_atom_types = 2
            self.label_to_atom_type = {i: a for i, a in enumerate(["C", "O"])}
        elif dataset_str_id.lower() == "qm9":
            self.dataset_str_id = dataset_str_id.lower()
            self.nr_atom_types = 5
            self.label_to_atom_type = {
                i: a for i, a in enumerate(["H", "C", "N", "O", "F"])
            }
            # atom charges ordered
            # H: 1
            # C: 6
            # N: 7
            # O: 8
            # F: 9
        else:
            raise ValueError(
                f"Dataset string identifyer '{dataset_str_id}' not known. Choose from 'CO2', 'QM9'."
            )

    def create_hash(self):
        """
        create hash string for prox function
        """
        r0, n, function_id = self.proximity_function.get_parameters()
        return f"f_{function_id}_r0_{r0}_n_{n}"

    def get_savefiles(self, savedir: str) -> dict:
        r0, n, function_id = self.proximity_function.get_parameters()
        return get_savefiles(
            savedir, r0, n, function_id, self.order, self.dataset_str_id
        )

    def get_savefiles_create_dirs(self, savedir: str) -> dict:
        savefiles = self.get_savefiles(savedir)
        os.makedirs(os.path.dirname(savefiles[1]), exist_ok=True)
        return savefiles
