

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json

# A sentinel object representing "no maintenance currently scheduled" (psi
# in the manuscript). Using a dedicated sentinel (rather than e.g. -1)
# keeps state keys unambiguous and easy to read when printed/serialized.
PSI = "psi"

State = Tuple[int, object, int]   # (i, tau, t)  where tau is an int or PSI


@dataclass
class CBMPInstance:
    """All parameters that define one instance of the CBMP problem."""

    m: int                         # highest (failed) deterioration level
    delta: int                     # maintenance delay
    T: int                         # planning horizon has T+1 decision epochs
    n: int                         # number of production levels (K = 0..n)
    B: float                       # production constant (Eq. 2)
    pi0: float                     # fixed shortage penalty
    pi1: float                     # variable shortage penalty (per unit)
    pi2: float                     # variable overproduction bonus (per unit)
    A: float                       # fixed cost of scheduling maintenance
    C_CM: float                    # corrective maintenance cost
    C_PM: float                    # preventive maintenance cost
    demand: Dict[int, float]       # {t: phi_t} for t = 1..T
    P: Dict[int, List[List[float]]]  # {K: (m+1)x(m+1) transition matrix}

    def omega(self, K: int) -> float:
        """Production rate at level K (Table 6: omega_K = K / n)."""
        return K / self.n

    def Z(self, K: int) -> float:
        """Production quantity Z_t at level K (Eq. 2): Z = omega_K * B."""
        return self.omega(K) * self.B

    def phi(self, t: int) -> float:
        """Demand of period t. Defined for t = 1..T; 0 for t = T+1."""
        return self.demand.get(t, 0.0)


def default_instance() -> CBMPInstance:
    """
    Parameters of the numerical example in Section 5 of the manuscript.
    Expected result (verification target): V(0, PSI, 1) = 1633
    (see manuscript, Table 19 / Section 5 discussion).
    """
    m = 3
    delta = 3
    T = 10
    n = 4
    B = 1000

    pi0, pi1, pi2 = 10, 0.5, 1
    A = 50
    C_CM = 2000
    C_PM = 200

    demand = {1: 500, 2: 700, 3: 100, 4: 1400, 5: 250,
              6: 200, 7: 500, 8: 180, 9: 850, 10: 1500}

    # Transition matrices P^K, K = 0..4 (Section 5).
    P = {
        0: [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]],
        1: [[0.5, 0.3, 0.15, 0.05],
            [0,   0.5, 0.3,  0.2],
            [0,   0,   0.5,  0.5],
            [0,   0,   0,    1]],
        2: [[0.3, 0.4, 0.2, 0.1],
            [0,   0.3, 0.4, 0.3],
            [0,   0,   0.3, 0.7],
            [0,   0,   0,   1]],
        3: [[0.15, 0.45, 0.25, 0.15],
            [0,    0.15, 0.5,  0.35],
            [0,    0,    0.15, 0.85],
            [0,    0,    0,    1]],
        4: [[0.05, 0.5,  0.3,  0.15],
            [0,    0.05, 0.55, 0.4],
            [0,    0,    0.05, 0.95],
            [0,    0,    0,    1]],
    }

    return CBMPInstance(m=m, delta=delta, T=T, n=n, B=B,
                         pi0=pi0, pi1=pi1, pi2=pi2, A=A,
                         C_CM=C_CM, C_PM=C_PM, demand=demand, P=P)


def extruder_instance() -> CBMPInstance:
    """
    Parameters of the application example in Section 6 of the manuscript
    (extruder / bottle-cap production system, delta = 2, n = 2, m = 4).
    """
    m = 4
    delta = 2
    T = 6
    n = 2
    B = 1000

    pi0, pi1, pi2 = 2, 0.1, 1
    A = 1000
    C_CM = 1000
    C_PM = 500

    demand = {1: 850, 2: 115, 3: 1500, 4: 1400, 5: 250, 6: 700}

    P = {
        0: [[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]],
        1: [[0.4, 0.3, 0.2, 0.05, 0],
            [0,   0.4, 0.3, 0.2,  0.1],
            [0,   0,   0.4, 0.4,  0.2],
            [0,   0,   0,   0.7,  0.3],
            [0,   0,   0,   0,    1]],
        2: [[0.1, 0.3, 0.4, 0.1, 0.1],
            [0,   0.1, 0.4, 0.3, 0.2],
            [0,   0,   0.1, 0.6, 0.3],
            [0,   0,   0,   0.2, 0.8],
            [0,   0,   0,   0,   1]],
    }

    return CBMPInstance(m=m, delta=delta, T=T, n=n, B=B,
                         pi0=pi0, pi1=pi1, pi2=pi2, A=A,
                         C_CM=C_CM, C_PM=C_PM, demand=demand, P=P)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    V: Dict[State, float] = field(default_factory=dict)
    policy: Dict[State, object] = field(default_factory=dict)

    def value(self, i: int, tau, t: int) -> Optional[float]:
        return self.V.get((i, tau, t))


class CBMPSolver:
    """
    Solves the finite-horizon CBMP MDP by backward induction, following
    Table 4 / Eq. 4-14 and the 8 scenarios of Table 3 in the manuscript.

    Every state is visited and solved exactly once (t = T+1 down to 1),
    consistent with the backward-induction procedure described in the
    manuscript's computational-complexity discussion (Section 5).
    """

    def __init__(self, inst: CBMPInstance):
        self.inst = inst
        self.V: Dict[State, float] = {}
        self.policy: Dict[State, object] = {}

    # -- shared helper terms ------------------------------------------------

    def _production_term(self, K: int, t: int) -> float:
        """
        pi0 * y_t + pi1 * max(0, phi_t - Z_t) - pi2 * max(0, Z_t - phi_t)
        (the bracketed cost/bonus term common to Eq. 4-6)
        """
        inst = self.inst
        Z = inst.Z(K)
        phi = inst.phi(t)
        y = 1 if phi > Z else 0
        shortage = max(0.0, phi - Z)
        surplus = max(0.0, Z - phi)
        return inst.pi0 * y + inst.pi1 * shortage - inst.pi2 * surplus

    def _transition_sum(self, i: int, K: int, tau_next, t_next: int) -> float:
        """sum_{j=i}^{m} P_ij^K * V(j, tau_next, t_next)   (Eq. 4-6)."""
        inst = self.inst
        total = 0.0
        row = inst.P[K][i]
        for j in range(i, inst.m + 1):
            v_next = self.V.get((j, tau_next, t_next))
            if v_next is None:
                raise KeyError(f"Missing V({j},{tau_next},{t_next}); "
                                f"backward induction order violated.")
            total += row[j] * v_next
        return total

    # -- scenario equations ---------------------------------------------------

    def _scenario_1(self, i: int, t: int) -> Tuple[float, object]:
        """Eq. 4: i != m, tau = PSI, T - t > delta."""
        inst = self.inst
        best_val = None
        best_act = None
        # Branch (I): adjust production only (no maintenance scheduled)
        for K in range(inst.n + 1):
            val = self._production_term(K, t) + self._transition_sum(i, K, PSI, t + 1)
            if best_val is None or val < best_val:
                best_val, best_act = val, ("produce_only", K)
        # Branch (II): adjust production AND schedule maintenance
        for K in range(inst.n + 1):
            val = (inst.A + self._production_term(K, t)
                   + self._transition_sum(i, K, inst.delta - 1, t + 1))
            if val < best_val:
                best_val, best_act = val, ("produce_and_schedule", K)
        return best_val, best_act

    def _scenario_2(self, i: int, t: int) -> Tuple[float, object]:
        """Eq. 5: i != m, tau = PSI, T - t <= delta."""
        inst = self.inst
        best_val = None
        best_act = None
        for K in range(inst.n + 1):
            val = self._production_term(K, t) + self._transition_sum(i, K, PSI, t + 1)
            if best_val is None or val < best_val:
                best_val, best_act = val, ("produce_only", K)
        return best_val, best_act

    def _scenario_3(self, i: int, tau: int, t: int) -> Tuple[float, object]:
        """Eq. 6: i != m, tau in {1,...,delta-1}."""
        inst = self.inst
        best_val = None
        best_act = None
        for K in range(inst.n + 1):
            val = self._production_term(K, t) + self._transition_sum(i, K, tau - 1, t + 1)
            if best_val is None or val < best_val:
                best_val, best_act = val, ("produce_only", K)
        return best_val, best_act

    def _scenario_4(self, tau: int, t: int) -> Tuple[float, object]:
        """Eq. 7: i == m, tau in {1,...,delta-1}. Production is zero;
        jumps directly to V(m, 0, t+tau)."""
        inst = self.inst
        penalty = tau * inst.pi0 + inst.pi1 * sum(inst.phi(j) for j in range(t, t + tau + 1))
        v_next = self.V[(inst.m, 0, t + tau)]
        return penalty + v_next, ("do_nothing",)

    def _scenario_5(self, i: int, t: int) -> Tuple[float, object]:
        """Eq. 8: tau == 0 (maintenance action arrives this period)."""
        inst = self.inst
        v_next = self.V[(0, PSI, t + 1)]
        if i == inst.m:
            val = inst.C_CM + inst.pi0 + inst.pi1 * inst.phi(t) + v_next
            act = ("CM",)
        elif 0 < i < inst.m:
            val = inst.C_PM + inst.pi0 + inst.pi1 * inst.phi(t) + v_next
            act = ("PM",)
        else:  # i == 0
            val = inst.pi0 + inst.pi1 * inst.phi(t) + v_next
            act = ("do_nothing",)
        return val, act

    def _scenario_6(self, t: int) -> Tuple[float, object]:
        """Eq. 9: i == m, tau == PSI, T - t > delta."""
        inst = self.inst
        schedule_val = (inst.A + inst.pi0 + inst.pi1 * inst.phi(t)
                         + self.V[(inst.m, inst.delta - 1, t + 1)])
        nothing_val = inst.pi0 + inst.pi1 * inst.phi(t) + self.V[(inst.m, PSI, t + 1)]
        if schedule_val <= nothing_val:
            return schedule_val, ("schedule_CM",)
        return nothing_val, ("do_nothing",)

    def _scenario_7(self, t: int) -> Tuple[float, object]:
        """Eq. 10: i == m, tau == PSI, T - t <= delta."""
        inst = self.inst
        penalty = inst.pi0 * (inst.T - t) + inst.pi1 * sum(
            inst.phi(j) for j in range(t, inst.T + 1))
        val = penalty + self.V[(inst.m, PSI, inst.T + 1)]
        return val, ("do_nothing",)

    def _scenario_8(self, i: int) -> Tuple[float, object]:
        """Eq. 11: terminal state t = T+1."""
        inst = self.inst
        if i == inst.m:
            return inst.C_CM, ("CM",)
        elif 0 < i < inst.m:
            return inst.C_PM, ("PM",)
        return 0.0, ("do_nothing",)

    # -- driver ---------------------------------------------------------------

    def solve(self) -> SolveResult:
        inst = self.inst
        m, delta, T = inst.m, inst.delta, inst.T

        # t = T+1 (terminal state, Eq. 11 / Scenario 8) -----------------------
        for i in range(m + 1):
            val, act = self._scenario_8(i)
            self.V[(i, PSI, T + 1)] = val
            self.policy[(i, PSI, T + 1)] = act

        # t = T, T-1, ..., 1 (backward induction) ------------------------------
        for t in range(T, 0, -1):
            # tau = 0 states valid only for t in {delta+1, ..., T} (Eq. 8 domain)
            if delta + 1 <= t <= T:
                for i in range(m + 1):
                    val, act = self._scenario_5(i, t)
                    self.V[(i, 0, t)] = val
                    self.policy[(i, 0, t)] = act

            # tau in {1, ..., delta-1}
            # Only reachable/well-defined when the scheduled maintenance
            # arrival epoch (t + tau) falls within {delta+1, ..., T}, i.e.
            # the successor state (., 0, t+tau) is covered by Scenario 5's
            # domain (Eq. 8). Combined with "scheduling only for t < T -
            # delta" (Scenario 1), this matches exactly the states left
            # blank ("-") in the manuscript's result tables (Tables 8-18).
            for tau in range(1, delta):
                # Reachability guards:
                #   (a) arrival epoch t+tau must not exceed T (successor
                #       tau=0 state must satisfy Scenario 5's domain);
                #   (b) the implied scheduling epoch t-(delta-tau) must be
                #       >= 1 (maintenance cannot have been scheduled before
                #       the start of the planning horizon).
                if t + tau > T or (t - (delta - tau)) < 1:
                    continue
                for i in range(m):  # i != m
                    val, act = self._scenario_3(i, tau, t)
                    self.V[(i, tau, t)] = val
                    self.policy[(i, tau, t)] = act
                # i == m
                val, act = self._scenario_4(tau, t)
                self.V[(m, tau, t)] = val
                self.policy[(m, tau, t)] = act

            # tau = PSI
            for i in range(m):  # i != m
                if T - t > delta:
                    val, act = self._scenario_1(i, t)
                else:
                    val, act = self._scenario_2(i, t)
                self.V[(i, PSI, t)] = val
                self.policy[(i, PSI, t)] = act
            # i == m
            if T - t > delta:
                val, act = self._scenario_6(t)
            else:
                val, act = self._scenario_7(t)
            self.V[(m, PSI, t)] = val
            self.policy[(m, PSI, t)] = act

        return SolveResult(V=self.V, policy=self.policy)

