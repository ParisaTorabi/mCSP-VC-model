from contextlib import redirect_stdout
import os
import json
import numpy as np
from scipy.spatial import distance_matrix

from MILP_model.make_model import make_model


def load_instance(file_path: str) -> dict:
    with open(file_path) as f:
        data = json.loads(f.read())

    data["C"] = np.arange(1, data["n"] + 1).tolist()
    data["D"] = distance_matrix(data["locations"], data["locations"])
    data["TG"] = data["D"] / data["SV"]
    data["TD"] = data["D"] / data["SD"]
    return data


if __name__ == "__main__":
    # instance set and the instances (1-80 for set 1 and 1-25 for set 2) to be given
    data_set = 1
    ins_nums = range(1, 81)
    # the folders for logs and solutions of the MILP
    log_dir = f".\\Logs\\set_{data_set}\\"
    sol_dir = f".\\Solutions\\set_{data_set}\\"
    # time limit given to the solver
    time_limit = 10000
    for ins_num in ins_nums:
        instance = load_instance(f".\\set_{data_set}\\instance_{ins_num:03}.json")

        m = make_model(instance=instance)
        if not os.path.exists(f"{log_dir}{int(time_limit//1000)}KS\\"):
            os.makedirs(f"{log_dir}{int(time_limit//1000)}KS\\")
        with open(
            f"{log_dir}{int(time_limit//1000)}KS\\instance_{ins_num:03}_logs.txt",
            "w",
        ) as log_file:
            with redirect_stdout(log_file):
                m.setParam("TimeLimit", time_limit)
                m.optimize()
                print("-----------------------------------")
        if m.SolCount:
            if not os.path.exists(f"{sol_dir}{int(time_limit//1000)}KS\\"):
                os.makedirs(f"{sol_dir}{int(time_limit//1000)}KS\\")
            m.write(f"{sol_dir}{int(time_limit//1000)}KS\\ins_{ins_num:03}.sol")
