import numpy as np
from typing import List, Tuple
import itertools

from petsc4py import PETSc
from slepc4py import SLEPc

from .differential_equation import DifferentialEquation

def _check_eigen_triplets(diff_eq, V, W, L):
    
    r = V.getActiveColumns()[-1]

    # Biorthogonality check: diag(W^H V) ≈ 1
    WtV = V.dot(W)
    assert np.linalg.norm(WtV.getDenseArray() - np.eye(r)) <= 1e-10
    WtV.destroy()

    # Eigenvalue check
    AV = V.duplicate()
    for i in range (r):
        v = V.getColumn(i)
        Av = AV.getColumn(i)
        Av = diff_eq.evaluate_linear_term(v, Av)
        AV.restoreColumn(i, Av)
        V.restoreColumn(i, v)
    WtAV = AV.dot(W)
    assert np.linalg.norm(WtAV.getDenseArray() - np.diag(L)) <= 1e-10
    WtAV.destroy()


class SpectralSubmanifold:
    r"""
    Compute a spectral submanifold :math:`P(s)` and the
    intrinsic dynamics :math:`\dot{s} = \Lambda s + g(s)`.

    :param diff_eq: differential equation providing
        :math:`A`, :math:`B(q,q)`, and eigenvalues
    :type diff_eq: DifferentialEquation
    :param r: dimension of the latent space
    :type r: int
    :param m: polynomial expansion order
    :type m: int
    :param conj_to_linear_dynamics: if True, set :math:`g(s) = 0`
        (conjugacy to linear dynamics); if False, solve for
        nonlinear :math:`g(s)`
    :type conj_to_linear_dynamics: bool
    """

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # --------- Initialization    ---------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def __init__(
        self, diff_eq: DifferentialEquation, r: int, m: int,
        conj_to_linear_dynamics: bool = False,
    ) -> None:

        self.diff_eq = diff_eq
        self.r = r
        self.m = m
        self.conj_to_linear_dynamics = conj_to_linear_dynamics

        if self.diff_eq.get_poly_degree() > 2:
            raise ValueError("Only quadratic dynamics are supported for now.")

        self.ps = None
        self.gs = None
        self.Lams = None
        self.V = None
        self.W = None

        self.ssm_multiindices = self.compute_multiindices(self.m)
        self.ssm_quad_rhs_idc = self.compute_quadratic_rhs_combinations(self.m)
        self.ssm_nonlin_dynmc = self.compute_nonlinear_dynamics_combinations(
            self.m
        )

    def compute_multiindices(self, m: int) -> List[Tuple[int, ...]]:
        r"""
        Compute all multi-indices :math:`j` with :math:`r` entries
        such that :math:`0 \leq |j| \leq m`. For example, with
        :math:`r = 2` and :math:`m = 2`:
        :code:`[(0,0), (0,1), (1,0), (0,2), (1,1), (2,0)]`.

        :param m: maximum degree
        :type m: int

        :rtype: List[Tuple[int, ...]]
        """
        lst = [
            self._generate_constrained_compositions(i, m)[::-1]
            for i in range(m + 1)
        ]
        return list(itertools.chain.from_iterable(lst))

    def compute_quadratic_rhs_combinations(
        self, m: int
    ) -> List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
        r"""
        Compute multi-index pair combinations for the quadratic
        right-hand side of the SSM equations.

        :param m: maximum degree
        :type m: int

        :rtype: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
        """
        return [
            self._generate_quadratic_pairs(j, m) for j in self.ssm_multiindices
        ]

    def compute_nonlinear_dynamics_combinations(
        self, m: int
    ) -> List[List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]]:
        r"""
        Compute multi-index pair combinations for the interaction
        between the nonlinear dynamics :math:`g(s)` and
        :math:`D_s P(s)`.

        :param m: maximum degree
        :type m: int

        :rtype: List[List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]]
        """
        return [
            self._generate_nonlinear_dynamics_pairs(j, m)
            for j in self.ssm_multiindices
        ]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # --------- Helper functions ----------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _get_multiindex_index(
        self, multiindex: List[Tuple[int, ...]], j: Tuple[int, ...]
    ) -> int:
        r"""Return the position of multi-index :math:`j` in the list."""
        return multiindex.index(j)

    def _generate_constrained_compositions(
        self, l: int, m: int
    ) -> List[Tuple[int, ...]]:
        r"""
        Generate all multi-indices :math:`(j_1, \ldots, j_r)` with
        :math:`0 \leq j_k \leq m` and :math:`\sum_k j_k = l`.

        :param l: target sum of the multi-index entries
        :type l: int
        :param m: maximum value per entry
        :type m: int

        :rtype: List[Tuple[int, ...]]
        """

        results = []

        def backtrack(current_tuple):
            # Check if the current tuple has the desired number of elements r
            if len(current_tuple) == self.r:
                # Check if the sum of all elements is equal to l. If so,
                # append the current tuple to the results list and break out
                # of this backtrack() call. Otherwise, just break out of
                # the current call.
                if sum(current_tuple) == l:
                    results.append(tuple(current_tuple))
                return

            # Compute the maximum value max_val that is attainable by the new
            # entry of the tuple, and loop over all values from 0 to
            # max_val + 1 to explore candidates
            max_val = min(m, l - sum(current_tuple))
            for val in range(max_val + 1):
                backtrack(current_tuple + [val])

        backtrack([])
        return results

    def _generate_quadratic_pairs(
        self, j: Tuple[int, ...], m: int
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        r"""
        Generate all non-trivial pairs :math:`(i, l)` with
        :math:`i + l = j` and :math:`0 \leq i_k, l_k \leq m`.

        :param j: target multi-index
        :type j: Tuple[int, ...]
        :param m: maximum value per entry
        :type m: int

        :rtype: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        """

        # For each element jk of the target multiindex j, compute
        # all possible multiindices whose sum adds to jk.
        component_decomposition_options = []
        for jk in j:
            valid_pairs = []
            for ik in range(jk + 1):
                lk = jk - ik
                valid_pairs.append((ik, lk)) if 0 <= lk <= m else None
            component_decomposition_options.append(valid_pairs)
        # Compute their cartesian product. The list all_combined_pairs
        # contains tuples, each containing two tuples of candidate products.
        all_combined_pairs = list(
            itertools.product(*component_decomposition_options)
        )

        zero_tuple = tuple([0] * len(j))
        final_pairs = []
        for combined_pair in all_combined_pairs:
            i_tuple = tuple(item[0] for item in combined_pair)
            l_tuple = tuple(item[1] for item in combined_pair)
            # Exclude trivial combinations
            if i_tuple != zero_tuple and l_tuple != zero_tuple:
                final_pairs.append((i_tuple, l_tuple))

        return final_pairs

    def _generate_nonlinear_dynamics_pairs(
        self, j: Tuple[int, ...], m: int
    ) -> List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
        r"""
        Generate multi-index pairs for the :math:`D_s P(s) \, g(s)`
        interaction terms at multi-index :math:`j`.

        :param j: target multi-index
        :type j: Tuple[int, ...]
        :param m: maximum value per entry
        :type m: int

        :rtype: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
        """
        list_of_pairs = []
        zero_tuple = tuple([0] * len(j))
        for k in range(len(j)):
            # For each element jk of the target multiindex j, compute
            # all possible multiindices whose sum adds to jk.
            component_decomposition_options = []
            for idx, jk in enumerate(j):
                valid_pairs = []
                end = jk + 2 if idx == k else jk + 1
                for ik in range(end):
                    lk = (end - 1) - ik
                    valid_pairs.append((ik, lk)) if 0 <= lk <= m else None
                component_decomposition_options.append(valid_pairs)
            # Compute their cartesian product. The list all_combined_pairs
            # contains tuples, each containing two tuples of candidate products.
            all_combined_pairs = list(
                itertools.product(*component_decomposition_options)
            )

            final_pairs = []
            for combined_pair in all_combined_pairs:
                i_tuple = tuple(item[0] for item in combined_pair)
                l_tuple = tuple(item[1] for item in combined_pair)
                # Exclude trivial combinations and combinations involving
                # g_j corresponding to terms of zeroth and first order.
                # (remember that g(s) is second order or higher). Also,
                # exclude terms involving p_{(1,0,...)}, p_{(0,1,...)}, etc.
                # which are handled explicitly outside this function
                if (
                    i_tuple != zero_tuple
                    and l_tuple != zero_tuple
                    and sum(i_tuple) > 1
                    and sum(l_tuple) > 1
                ):
                    if i_tuple[k] == 0:
                        continue
                    else:
                        i_tuple_ = list(i_tuple)
                        i_tuple_[k] = i_tuple[k] - 1
                        ipl = tuple(
                            map(lambda x, y: x + y, i_tuple_, list(l_tuple))
                        )
                        if ipl == j:
                            final_pairs.append((i_tuple, l_tuple))
            list_of_pairs.append(final_pairs)

        return list_of_pairs

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # --------- Public functions ---------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def solve(
        self,
        Phi: SLEPc.BV,
        Psi: SLEPc.BV,
        Lams: np.ndarray,
        scaling: float = 1.0,
    ) -> Tuple[List[PETSc.Vec], List[np.ndarray]]:
        r"""
        Compute the coefficients :math:`p_j` and :math:`g_j` in the
        polynomial expansions of :math:`P(s)` and :math:`g(s)`.

        :param Phi: right eigenvectors
        :type Phi: SLEPc.BV
        :param Psi: left eigenvectors
        :type Psi: SLEPc.BV
        :param Lams: eigenvalues of shape :math:`(r,)`
        :type Lams: np.ndarray
        :param scaling: scaling applied to eigenvectors
        :type scaling: float

        :return: manifold coefficients :math:`p_j` and
            dynamics coefficients :math:`g_j`
        :rtype: (List[PETSc.Vec], List[np.ndarray])
        """
        r = len(Lams)
        conj = self.conj_to_linear_dynamics

        # Scale eigenvectors: V = Phi * scaling,  W = Psi / scaling
        V = Phi.copy()
        W = Psi.copy()
        objs = [V, W]
        factors = [scaling, 1 / scaling]
        for (j, obj) in enumerate(objs):
            obj_mat = obj.getMat()
            obj_mat.scale(factors[j])
            obj.restoreMat(obj_mat)
        
        _check_eigen_triplets(self.diff_eq, V, W, Lams)

        # Get a template vector for creating new PETSc Vecs
        v_ref = V.getColumn(0)
        template = v_ref.duplicate()
        V.restoreColumn(0, v_ref)

        # Zeroth and first order terms in the manifold expansion
        p0 = template.duplicate()
        p0.zeroEntries()
        ps: List[PETSc.Vec] = [p0]
        for k in range(r):
            v_k = V.getColumn(k)
            ps.append(v_k.copy())
            V.restoreColumn(k, v_k)

        # Zeroth and first order terms in g(s), internal dynamics
        gs: List[np.ndarray] = [np.zeros(r, dtype=complex)] * (r + 1)

        # Reusable work vector
        rhs = template.duplicate()
        rhsj = template.duplicate()

        for j_idx in range(r + 1, len(self.ssm_multiindices)):
            j = self.ssm_multiindices[j_idx]
            shift = np.dot(Lams, np.asarray(j))
            rhs.zeroEntries()

            # Compute contrib. from quadratic nature of the governing equations
            for pair in self.ssm_quad_rhs_idc[j_idx]:
                idces = [
                    self._get_multiindex_index(self.ssm_multiindices, p)
                    for p in pair
                ]
                rhsj = self.diff_eq.evaluate_quadratic_term(
                    ps[idces[0]], ps[idces[1]], rhsj
                )
                rhs.axpy(1.0, rhsj)

            if not conj:
                # Compute contrib. from nonlinear latent-space dynamics
                for k in range(len(self.ssm_nonlin_dynmc[j_idx])):
                    for pair in self.ssm_nonlin_dynmc[j_idx][k]:
                        idxp = self._get_multiindex_index(
                            self.ssm_multiindices, pair[0]
                        )
                        idxg = self._get_multiindex_index(
                            self.ssm_multiindices, pair[1]
                        )
                        rhs.axpy(-pair[0][k] * gs[idxg][k], ps[idxp])

            # Subtract V @ gj from rhs, then solve (shift*I - A) pj = rhs
            gj = W.dotVec(rhs) if not conj else np.zeros(r, dtype=complex)
            V.multVec(-1.0, 1.0, rhs, gj)  # rhs = -V @ gj + rhs
            pj = self.diff_eq.solve_linear_system(shift, rhs)

            if not conj:
                proj = W.dotVec(pj)
                if np.linalg.norm(proj) >= 1e-10:
                    raise ValueError (
                        f"|W^* pj| > tolerance. Please try modifying "
                        f"the scaling parameters when running .solve()"
                    )
            gs.append(gj)
            ps.append(pj)

        template.destroy()

        self.ps = ps
        self.gs = gs
        self.Lams = Lams
        self.V = V
        self.W = W
        return ps, gs

    def decode(self, s: np.ndarray) -> PETSc.Vec:
        r"""
        Evaluate :math:`P(s)` for a given latent-space state.

        :param s: latent-space coordinates of shape :math:`(r,)`
        :type s: np.ndarray

        :rtype: PETSc.Vec
        """
        if self.ps is None:
            raise RuntimeError("Call solve() first.")
        vec = self.ps[0].duplicate()
        vec.zeroEntries()
        for idx, j in enumerate(self.ssm_multiindices):
            vec.axpy(np.prod(s ** np.asarray(j)), self.ps[idx])
        return vec

    def encode(self, q: PETSc.Vec) -> np.ndarray:
        r"""
        Project a full-state vector onto the latent space
        via :math:`W^* q`.

        :param q: full-state vector
        :type q: PETSc.Vec

        :rtype: np.ndarray
        """
        if self.W is None:
            raise RuntimeError("Call solve() first.")
        return self.W.dotVec(q)

    def latent_space_dynamics(
        self, t: float, s: np.ndarray
    ) -> np.ndarray:
        r"""
        Compute :math:`\dot{s} = \Lambda s + g(s)`.

        :param t: time (unused, included for ODE-solver compatibility)
        :type t: float
        :param s: latent-space coordinates of shape :math:`(r,)`
        :type s: np.ndarray

        :rtype: np.ndarray
        """
        if self.Lams is None:
            raise RuntimeError("Call solve() first.")
        ds = self.Lams * s
        if not self.conj_to_linear_dynamics:
            _start = len(self.Lams) + 1
            J = np.array(self.ssm_multiindices[_start:])
            G = np.array(self.gs[_start:])
            monomials = np.prod(s[None, :] ** J, axis=1)
            ds = ds + monomials @ G

        return ds

    def estimate_convergence_radius(
        self,
    ) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
        r"""
        Estimate the convergence radius of the SSM polynomial
        expansion by fitting a line to
        :math:`(k,\,\log_{10}(C_k))`, where

        .. math::

            C_k = \sum_{|j|=k} \lVert p_j \rVert_1.

        Uses the Theil-Sen estimator for robustness to
        near-resonance outliers.

        :return: :math:`(R, \text{orders}, C_k, \text{slope},
            \text{intercept})`
        :rtype: (float, np.ndarray, np.ndarray, float, float)
        :raises ValueError: if fewer than 2 valid points are
            available for the fit
        """
        if self.ps is None:
            raise RuntimeError("Call solve() first.")
        from collections import defaultdict

        order_sums = defaultdict(float)
        for idx, j in enumerate(self.ssm_multiindices):
            k = sum(j)
            if k == 0:
                continue  # constant term is zero by construction
            order_sums[k] += self.ps[idx].norm(PETSc.NormType.NORM_1)

        orders = np.array(sorted(order_sums.keys()))
        coeff_sums = np.array([order_sums[k] for k in orders])

        valid = coeff_sums > 0
        log_sums = np.full_like(coeff_sums, np.nan)
        log_sums[valid] = np.log10(coeff_sums[valid])

        from scipy.stats import theilslopes

        fit_orders = orders[valid]
        fit_logs = log_sums[valid]

        # Omit first 33% of data points (low-order terms may not be asymptotic)
        skip = max(1, len(fit_orders) // 3)
        fit_orders = fit_orders[skip:]
        fit_logs = fit_logs[skip:]

        if fit_orders.size < 2:
            raise ValueError("Need at least 2 valid points to fit a line.")

        # Theil-Sen estimator: median of pairwise slopes, robust to outliers
        # (unlike np.polyfit/OLS, which is skewed by near-resonance artifacts)
        slope, intercept, _, _ = theilslopes(fit_logs, fit_orders)
        R_estimate = 10 ** (-slope)

        return R_estimate, orders, coeff_sums, slope, intercept