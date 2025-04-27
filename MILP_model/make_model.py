import gurobipy as gp
from gurobipy import GRB, tuplelist
import numpy as np


def make_model(instance: dict) -> gp.Model:
    n = instance["n"]
    m = gp.Model("mCTPVC")

    M1 = n - 1
    M2 = instance["ND"]
    M3 = instance["B"]
    M4 = (
        instance["B"]
        + instance["U"] * np.max(instance["TD"])
        + instance["Uh"] * np.sum(instance["TD"])
    )
    M5 = instance["U"] * np.max(instance["TD"])
    M6 = max(0, (instance["U"] * np.max(instance["TD"]) - instance["B"]))
    M7 = (
        np.sum(instance["TG"])
        + np.max(instance["TG"])
        + np.sum(instance["TD"])
        + np.max(instance["TD"])
    )

    M9 = n * instance["ND"]
    M10 = np.max(instance["D"])
    M11 = n * (n - 1)
    # M12 = instance["theta"]

    # print(f"M variables defined after {time.time()-time_1} seconds ")

    # M = M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = M10 = M11 = M12 = 1000000
    V = set(instance["C"] + [n + 1])
    C = set(instance["C"])
    # VG = set(instance["CG"] + [instance["n"] + 1])
    K = set(np.arange(1, n))
    A = tuplelist([(i, j) for i in V for j in V if i != j])
    # AG = tuplelist([(i, j) for i in VG for j in VG if i != j])

    # print(f"Sets defined after {time.time()-time_1} seconds")

    # binary variables
    x = m.addVars(A, K, vtype=GRB.BINARY, name="x")
    y = m.addVars(A, vtype=GRB.BINARY, name="y")
    zG = m.addVars(V, vtype=GRB.BINARY, name="zG")
    zD = m.addVars(V, K, vtype=GRB.BINARY, name="zD")
    zC = m.addVars(V, vtype=GRB.BINARY, name="zC")
    s = m.addVars(A, vtype=GRB.BINARY, name="s")
    hL = m.addVars(V, K, vtype=GRB.BINARY, name="hL")
    hR = m.addVars(V, K, vtype=GRB.BINARY, name="hR")

    # mt = m.addVar(vtype=GRB.BINARY, name="mt")
    # integer
    d = m.addVars(V, ub=instance["ND"], vtype=GRB.INTEGER, name="d")

    # continuous

    tGA = m.addVars(V, name="tGA")
    tGL = m.addVars(V, name="tGL")
    tDA = m.addVars(V, name="tDA")
    tDL = m.addVars(V, name="tDL")
    pD = m.addVars(V, ub=instance["B"], name="pD")
    r = m.addVars(C, ub=instance["theta"], name="r")
    max_time = m.addVar()

    # print(f"Variables defined after {time.time()-time_1} seconds")

    # m.addConstr(max_time == mt * tGA[n + 1] + (1 - mt) * tDA[n + 1], name="max_time")
    # m.addConstr(tGA[n + 1] >= tDA[n + 1] - M * (1 - mt), "mt1")
    # m.addConstr(tGA[n + 1] <= tDA[n + 1] - M * mt, "mt2")
    m.addConstr(max_time >= tGA[n + 1], name="max_time_1")
    m.addConstr(max_time >= tDA[n + 1], name="max_time_2")
    m.setObjective(max_time, GRB.MINIMIZE)
    #
    # print(f"Objective function set after {time.time()-time_1} seconds")

    m.addConstrs(((gp.quicksum(zD[i, k] for k in K)) + zC[i] + zG[i] == 1 for i in V), name="C2")

    # m.addConstrs(((gp.quicksum(zD[i, k] for k in K)) + zG[i] <= 1 for i in V), name="C2_1")

    # m.addConstrs((zG[i] == 0 for i in instance["CD"]), name="C3")
    m.addConstrs(
        ((gp.quicksum(x[j, i, k] for j in V if (j, i) in A)) >= zD[i, k] for i in V for k in K),
        name="C3",
    )

    m.addConstrs((zG[i] == gp.quicksum(y[j, i] for j in V if (j, i) in A) for i in V), name="C4")

    # print(f"Up to C5 set after {time.time()-time_1} seconds")

    m.addConstrs(
        (
            hR[i, k] + gp.quicksum(x[i, j, k] for j in V if (i, j) in A)
            == hL[i, k] + gp.quicksum(x[j, i, k] for j in V if (j, i) in A)
            for i in C
            for k in K
        ),
        name="C5",
    )

    m.addConstrs(
        (gp.quicksum(x[i, j, k] for j in V if (i, j) in A) <= 1 for i in V for k in K), name="C6"
    )

    m.addConstrs(
        (gp.quicksum(x[j, i, k] for j in V if (j, i) in A) <= 1 for i in V for k in K), name="C7"
    )

    m.addConstrs(
        (
            gp.quicksum(y[i, j] for j in V if (i, j) in A)
            == gp.quicksum(y[j, i] for j in V if (j, i) in A)
            for i in V
        ),
        name="C8",
    )

    m.addConstr((gp.quicksum(y[n + 1, j] for j in V if (n + 1, j) in A) == 1), name="C9_1")
    m.addConstr((gp.quicksum(y[j, n + 1] for j in V if (j, n + 1) in A) == 1), name="C9_2")

    m.addConstrs(
        (
            gp.quicksum(x[i, j, k] for k in K) + gp.quicksum(x[j, i, k] for k in K) <= 1
            for (i, j) in A
        ),
        name="C10",
    )

    m.addConstrs((zD[i, k] + zD[j, k] >= x[i, j, k] for (i, j) in A for k in K), name="C11")

    m.addConstrs(
        (hL[i, k] >= x[i, j, k] + zD[j, k] - zD[i, k] - 1 for (i, j) in A for k in K), name="C12"
    )

    m.addConstrs(
        (hR[j, k] >= x[i, j, k] + zD[i, k] - zD[j, k] - 1 for (i, j) in A for k in K), name="C13"
    )

    m.addConstrs((gp.quicksum(hL[i, k] for k in K) <= M1 * zG[i] for i in V), name="C14")

    m.addConstrs((gp.quicksum(hR[i, k] for k in K) <= M1 * zG[i] for i in V), name="C15")

    m.addConstrs((hR[i, k] + hL[i, k] <= 1 for i in V for k in K), name="C16")
    m.addConstrs(
        (gp.quicksum(hL[i, k] for i in V) == gp.quicksum(hR[i, k] for i in V) for k in K),
        name="C17_1",
    )

    m.addConstrs((gp.quicksum(hR[i, k] for i in V) <= 1 for k in K), name="C17_2")

    # drones and battery

    m.addConstr(
        (
            d[n + 1] + gp.quicksum(x[n + 1, j, k] for j in V if (n + 1, j) in A for k in K)
            == instance["ND"]
        ),
        name="C18",
    )

    m.addConstrs(
        (
            d[i] + gp.quicksum(x[lr, j, k] for lr in V if (lr, j) in A for k in K)
            >= d[j]
            + gp.quicksum(x[j, lr, k] for lr in V if (j, lr) in A for k in K)
            - M2 * (1 - y[i, j])
            for (i, j) in A
        ),
        name="C19",
    )

    m.addConstrs(
        (
            d[i] + gp.quicksum(x[j, n + 1, k] for j in V if (j, n + 1) in A for k in K)
            >= instance["ND"] - M2 * (1 - y[i, n + 1])
            for i in C
        ),
        name="C20",
    )

    m.addConstrs((pD[i] >= instance["B"] - M3 * (1 - zG[i]) for i in V), name="C21")

    m.addConstrs(
        (
            pD[j]
            <= pD[i]
            - instance["U"] * instance["TD"][i - 1, j - 1]
            - instance["Uh"] * (tDL[j] - tDA[j])
            + M4 * (3 - x[i, j, k] - zD[i, k] - zD[j, k])
            for (i, j) in A
            for k in K
        ),
        name="C22",
    )

    m.addConstrs(
        (
            pD[j]
            <= pD[i]
            - instance["U"] * instance["TD"][i - 1, j - 1]
            - instance["Uh"] * (tDL[j] - tDA[j])
            + M4 * (3 - x[i, j, k] - hL[i, k] - zD[j, k])
            for (i, j) in A
            for k in K
        ),
        name="C23",
    )

    m.addConstrs(
        (
            pD[i] >= instance["U"] * instance["TD"][i - 1, j - 1] - M5 * (2 - x[i, j, k] - hR[j, k])
            for (i, j) in A
            for k in K
        ),
        name="C24",
    )

    m.addConstrs(
        (
            instance["B"]
            >= instance["U"] * instance["TD"][i - 1, j - 1] - M6 * (2 - x[i, j, k] - hL[i, k])
            for (i, j) in A
            for k in K
        ),
        name="C25",
    )

    # Time constraints

    m.addConstrs(
        (tGA[j] >= tGL[i] + instance["TG"][i - 1, j - 1] - M7 * (1 - y[i, j]) for (i, j) in A),
        name="C26",
    )

    # m.addConstrs(
    #     (tGA[j] <= tGL[i] + instance["TG"][i - 1, j - 1] + M7 * (1 - y[i, j]) for (i, j) in AG),
    #     name="C32",
    # )

    m.addConstrs((tGL[i] >= tGA[i] - M7 * (1 - zG[i]) for i in C), name="C27")

    m.addConstrs((tGL[i] >= tDA[i] - M7 * (1 - hR[i, k]) for i in C for k in K), name="C28")

    m.addConstrs(
        (
            tDA[j]
            >= tDL[i] + instance["TD"][i - 1, j - 1] - M7 * (1 - gp.quicksum(x[i, j, k] for k in K))
            for (i, j) in A
        ),
        name="C29",
    )

    m.addConstrs(
        (tDL[i] >= tDA[i] - M7 * (1 - gp.quicksum(zD[i, k] for k in K)) for i in C), name="C30"
    )

    m.addConstrs((tDL[i] >= tGL[i] - M7 * (1 - hL[i, k]) for i in V for k in K), name="C31")

    m.addConstrs((tDL[i] <= tGL[i] + M7 * (1 - hL[i, k]) for i in V for k in K), name="C32")

    m.addConstrs(
        (
            M9 * (gp.quicksum(zD[a, k] for a in C if a <= i))
            >= (gp.quicksum(zD[a, b] for a in C if a <= i for b in K if b >= k))
            for i in C
            for k in K
        ),
        name="C33",
    )

    # print(f"Up to C39 set after {time.time()-time_1} seconds")
    # Now the radius!

    m.addConstrs(
        (
            r[j]
            <= instance["alpha"] * (tDL[j] - tDA[j])
            + instance["beta"]
            + M7 * (1 - gp.quicksum(zD[j, k] for k in K))
            for j in C
        ),
        name="C34",
    )

    # m.addConstrs(
    #     (
    #         r[j]
    #         >= instance["alpha"] * (tDL[j] - tDA[j])
    #         + instance["beta"]
    #         - M12 * (1 - gp.quicksum(zD[j, k] for k in K))
    #         for j in C
    #     ),
    #     name="C41",
    # )
    m.addConstrs(
        (
            r[j] <= instance["alpha"] * (tGL[j] - tGA[j]) + instance["beta"] + M7 * (1 - zG[j])
            for j in C
        ),
        name="C35",
    )

    # m.addConstrs(
    #     (
    #         r[j] >= instance["alpha"] * (tGL[j] - tGA[j]) + instance["beta"] - M12 * (1 - zG[j])
    #         for j in instance["CG"]
    #     ),
    #     name="C43",
    # )

    m.addConstrs(
        (
            r[j] >= instance["D"][i - 1, j - 1] - M10 * (1 - s[i, j])
            for j in C
            for i in V
            if (i, j) in A
        ),
        name="C36",
    )

    m.addConstrs(
        (s[i, j] <= gp.quicksum(zD[j, k] for k in K) + zG[j] for j in C for i in V if (i, j) in A),
        name="C37",
    )

    # m.addConstrs(
    #     (r[j] <= M12 * gp.quicksum(s[i, j] for i in V if (i, j) in A) for j in C), name="C46"
    # )

    m.addConstrs(
        ((gp.quicksum(s[i, j] for j in C if (i, j) in A)) >= 1 - M11 * (1 - zC[i]) for i in C),
        name="C38",
    )

    m.addConstrs(
        ((gp.quicksum(s[i, j] for j in C if (i, j) in A)) <= M11 * zC[i] for i in C), name="C39"
    )

    return m
