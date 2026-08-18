
from cbmp_mdp import default_instance

inst = default_instance()
m, T, n = inst.m, inst.T, inst.n

V = {}

def prod_term(K, t):
    Z = inst.Z(K)
    phi = inst.phi(t)
    y = 1 if phi > Z else 0
    shortage = max(0.0, phi - Z)
    surplus = max(0.0, Z - phi)
    return inst.pi0 * y + inst.pi1 * shortage - inst.pi2 * surplus

def trans_sum(i, K, t_next):
    row = inst.P[K][i]
    return sum(row[j] * V[(j, t_next)] for j in range(i, m + 1))

# terminal, Eq. 14
for i in range(m + 1):
    if i == m:
        V[(i, T + 1)] = inst.C_CM
    elif 0 < i < m:
        V[(i, T + 1)] = inst.C_PM
    else:
        V[(i, T + 1)] = 0.0

for t in range(T, 0, -1):
    # i == m, Eq. 13 (valid for t in {2,...,T}; for t=1 the system starts
    # as-good-as-new so i=m is unreachable, but we compute anyway if needed)
    schedule_val = inst.A + inst.C_CM + inst.pi0 + inst.pi1 * inst.phi(t) + V[(0, t + 1)]
    nothing_val = inst.pi0 + inst.pi1 * inst.phi(t) + V[(m, t + 1)]
    V[(m, t)] = min(schedule_val, nothing_val)

    # i != m, Eq. 12
    for i in range(m):
        best = None
        for K in range(n + 1):
            val = prod_term(K, t) + trans_sum(i, K, t + 1)
            if best is None or val < best:
                best = val
        pm_val = inst.A + inst.C_PM + inst.pi0 + inst.pi1 * inst.phi(t) + V[(0, t + 1)]
        V[(i, t)] = min(best, pm_val)

