import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import itertools

from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from .differential_equation import DifferentialEquation
from ..utils.comms import (
    gather_vec_to_rank,
    scatter_vec_from_rank,
)


def _check_eigen_triplets(diff_eq, V, W, L, tol=1e-10):
    r = V.getActiveColumns()[-1]
    comm = diff_eq.get_comm()

    # Biorthogonality check: diag(W^H V) ≈ 1
    WtV = V.dot(W)
    err_bio = np.linalg.norm(WtV.getDenseArray() - np.eye(r))
    WtV.destroy()
    if err_bio > tol:
        from ..utils.miscellaneous import petscprint

        petscprint(
            comm,
            f"WARNING: biorthogonality error = {err_bio:.3e} "
            f"(tol = {tol:.0e})",
        )

    # Eigenvalue check
    AV = V.duplicate()
    for i in range(r):
        v = V.getColumn(i)
        Av = AV.getColumn(i)
        Av = diff_eq.evaluate_linear_term(0, v, Av)
        AV.restoreColumn(i, Av)
        V.restoreColumn(i, v)
    WtAV = AV.dot(W)
    err_eig = np.linalg.norm(WtAV.getDenseArray() - np.diag(L))
    WtAV.destroy()
    if err_eig > tol:
        from ..utils.miscellaneous import petscprint

        petscprint(
            comm,
            f"WARNING: eigenvalue error = {err_eig:.3e} (tol = {tol:.0e})",
        )


class SpectralSubmanifold:
    r"""
    Compute a spectral submanifold :math:`P(s)` and the
    intrinsic dynamics :math:`\dot{s} = \Lambda s + g(s)`.

    :param diff_eq: quadratic differential equation providing
        :math:`A`, :math:`B(q,q)`, and eigenvalues.
    :type diff_eq: DifferentialEquation
    :param r: dimension of the latent space.
    :type r: int
    :param m: polynomial expansion order.
    :type m: int
    :param latent_space_components: list of polynomial orders ``|j|``
        that the latent-space dynamics ``g(s)`` is allowed to contain.
        At every multi-index ``j`` with ``sum(j)`` *not* in this list,
        ``gj`` is forced to ``0`` AFTER its standard computation (and
        before being stored / used by subsequent orders), so the
        higher-order corrections are absorbed into the manifold map
        ``P(s)`` instead — a normal-form parametrisation.  ``None``
        (default) keeps every order, i.e. the natural parametrisation.
        ``latent_space_components = [1]`` reduces to the
        conjugacy-to-linear-dynamics case (``g(s) = Lams * s``) and is
        flagged internally as such.  Typical pitchfork-style example:
        ``latent_space_components = [1, 3]``.
    :type latent_space_components: Optional[List[int]]
    :param n_workers: number of MPI ranks that participate in the
        per-pair :meth:`DifferentialEquation.evaluate_quadratic_term`
        evaluation inside :meth:`solve`.  Default is ``world_size``
        (every rank works on the quadratic term).  Set
        ``1 <= n_workers <= world_size`` to opt-out a subset of
        ranks — useful when the per-call bilinear allocates so much
        memory that you cannot afford to run it on every rank.
        Workers are spread evenly across the rank space (decimation
        placement); non-workers still participate in all collectives
        (gathers + Allreduce) but skip the bilinear evaluation.

        This parallel path passes ``COMM_SELF`` Vecs to
        ``evaluate_quadratic_term``, so the user's subclass must
        tolerate that: e.g. it must not allocate workspace on a
        harmonic-balanced ``self._comm = COMM_WORLD`` BV.  Use
        ``n_workers=1`` to fall back to the bit-for-bit serial
        path.
    :type n_workers: Optional[int]
    """

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # --------- Initialization    ---------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def __init__(
        self,
        diff_eq: DifferentialEquation,
        r: int,
        m: int,
        latent_space_components: Optional[List[int]] = None,
        n_workers: Optional[int] = None,
    ) -> None:
        self.diff_eq = diff_eq
        self.r = r
        self.m = m
        # `conj_to_linear_dynamics` is the legacy "g(s) = Lams * s" flag,
        # now derived: True iff the only retained latent-space order is 1.
        self.latent_space_components = latent_space_components
        self.conj_to_linear_dynamics = latent_space_components == [1]

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

        # Build the worker pool and the per-rank routing table for the
        # parallel quadratic-RHS evaluation in solve().
        self._setup_worker_pool(n_workers)

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
    # --------- Worker-pool setup for parallel quadratic-RHS evaluation -------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _setup_worker_pool(self, n_workers: Optional[int]) -> None:
        r"""Decide which ``n_workers`` ranks evaluate
        :meth:`DifferentialEquation.evaluate_quadratic_term`, partition
        the pair lists across them, and pre-compute the routing table
        that drives the per-``j_idx`` gather in :meth:`solve`.

        Workers are placed by **decimation** across the world rank
        space (rank ``w * world_size // n_workers`` for
        ``w = 0..n_workers-1``).  On a typical MPI launch (adjacent
        ranks on the same node), this places at most one worker per
        node — balancing inbound bandwidth across nodes during the
        gather phase.

        Sets the following per-instance attributes:

        - ``self.n_workers`` (int): the active worker count.
        - ``self._world_comm`` (mpi4py.MPI.Comm): the diff_eq's world.
        - ``self._world_rank`` (int), ``self._world_size`` (int).
        - ``self._worker_world_ranks`` (List[int]): the chosen worker
          ranks in world-rank order.
        - ``self._worker_id`` (int): worker index in
          ``[0, n_workers)`` for worker ranks; ``-1`` otherwise.
        - ``self.my_ssm_quad_rhs_idc`` (List[List[pair]]): per-j_idx
          local pair list for this rank.  Empty list on non-workers.
        - ``self._needed_indices`` (List[Set[int]]): per-j_idx set of
          ``ps``-indices this rank needs in its pair loop.  Empty on
          non-workers.
        - ``self._workers_needing`` (List[Dict[int, List[int]]]):
          per-j_idx routing table — for each ``ps``-index, the list of
          world ranks that want it.  Identical on every rank (it's
          fully deterministic).
        """
        self._world_comm = self.diff_eq.get_comm().tompi4py()
        self._world_size = self._world_comm.Get_size()
        self._world_rank = self._world_comm.Get_rank()

        if n_workers is None:
            n_workers = self._world_size
        if not (1 <= n_workers <= self._world_size):
            raise ValueError(
                f"n_workers must satisfy 1 <= n_workers <= world_size; "
                f"got n_workers = {n_workers}, world_size = "
                f"{self._world_size}."
            )
        self.n_workers = n_workers

        # Decimation placement.
        self._worker_world_ranks = [
            w * self._world_size // n_workers for w in range(n_workers)
        ]
        rank_to_worker = {
            r: w for w, r in enumerate(self._worker_world_ranks)
        }
        self._worker_id = rank_to_worker.get(self._world_rank, -1)

        # Build per-j_idx local view + routing table.  All ranks run
        # this exact deterministic computation, so the tables agree by
        # construction (no MPI needed to sync them).
        self.my_ssm_quad_rhs_idc: List[List[Tuple]] = []
        self._needed_indices: List[Set[int]] = []
        self._workers_needing: List[Dict[int, List[int]]] = []

        for j_idx in range(len(self.ssm_multiindices)):
            full_pairs = self.ssm_quad_rhs_idc[j_idx]

            # Round-robin assignment of pairs to workers
            # (w_pairs[w] = pairs assigned to worker w).
            w_pairs = [
                full_pairs[w :: n_workers] for w in range(n_workers)
            ]
            # Union of ps-indices each worker needs.
            w_needs = [
                {
                    self._get_multiindex_index(self.ssm_multiindices, p)
                    for pair in w_pairs[w]
                    for p in pair
                }
                for w in range(n_workers)
            ]

            if self._worker_id >= 0:
                self.my_ssm_quad_rhs_idc.append(w_pairs[self._worker_id])
                self._needed_indices.append(w_needs[self._worker_id])
            else:
                self.my_ssm_quad_rhs_idc.append([])
                self._needed_indices.append(set())

            # Invert: ps_index -> sorted list of world ranks needing it.
            routing: Dict[int, List[int]] = {}
            for w, needs in enumerate(w_needs):
                dr = self._worker_world_ranks[w]
                for i in needs:
                    routing.setdefault(i, []).append(dr)
            self._workers_needing.append(routing)

    def _evaluate_quadratic_rhs_root(
        self,
        t: float,
        q1_vec: PETSc.Vec,
        q2_vec: PETSc.Vec,
        y_vec: PETSc.Vec,
    ) -> PETSc.Vec:
        r"""PETSc-Vec ↔ numpy adapter around
        :meth:`DifferentialEquation.evaluate_quadratic_term`.

        Three phases:

        1. Gather ``q1_vec`` and ``q2_vec`` to **rank 0 only** via
           :func:`gather_vec_to_rank` — non-root ranks contribute to
           the Gatherv but hold no copy of the full array.
        2. Rank 0 alone evaluates the bilinear in numpy.  No
           redundant compute across ranks; no per-rank memory blowup
           if the bilinear allocates large workspace.
        3. Scatter the rank-0 result back into the ownership ranges of
           ``y_vec`` via :func:`scatter_vec_from_rank` (the inverse
           of :func:`gather_vec_to_rank`).  Each rank receives only
           its local slice.

        This is what the serial (``n_workers == 1``) path uses to
        preserve the existing public behaviour of :meth:`solve` after
        switching the per-instant bilinear to its numpy signature."""
        q1_arr = gather_vec_to_rank(q1_vec, 0)
        q2_arr = gather_vec_to_rank(q2_vec, 0)

        Bqq_arr = None
        if self._world_rank == 0:
            Bqq_arr = self.diff_eq.evaluate_quadratic_term(
                t, q1_arr, q2_arr
            )

        return scatter_vec_from_rank(Bqq_arr, y_vec, 0)

    def _evaluate_quadratic_rhs_parallel(
        self,
        j_idx: int,
        ps: List[PETSc.Vec],
        rhs: PETSc.Vec,
    ) -> None:
        r"""Compute :math:`\sum_{(i_1, i_2) \in \mathrm{pairs}}
        B(p_{i_1}, p_{i_2})` in parallel and *replace* ``rhs`` with the
        result (caller must zero ``rhs`` beforehand).

        Four phases:

        1. **Gather**: for each ``ps[i]`` needed by any worker, do
           one :func:`gather_vec_to_rank` to the *primary* worker
           (``routing[i][0]``), which then forwards via blocking
           ``Send`` to each additional worker in the list.  Iterate
           in sorted index order so every rank issues the same
           sequence of collectives.
        2. **Local pair loop**: each worker iterates its assigned
           pairs, calling the numpy ``evaluate_quadratic_term`` on
           the cached arrays and accumulating into a local buffer.
        3. **Allreduce**: sum the per-worker buffers across
           ``world_comm`` so every rank holds the global RHS in numpy.
        4. **Inject**: each rank writes its ownership range of ``rhs``
           from the global numpy buffer.
        """
        N_global = rhs.getSize()
        routing = self._workers_needing[j_idx]

        # Phase 1: gather to primary worker + point-to-point forward
        # to any other workers that also need each ps[i].
        ps_cache: Dict[int, np.ndarray] = {}
        for i in sorted(routing.keys()):
            dest_ranks = routing[i]
            primary = dest_ranks[0]
            gathered = gather_vec_to_rank(ps[i], primary)

            secondary = dest_ranks[1:]
            if self._world_rank == primary:
                ps_cache[i] = gathered
                for dr in secondary:
                    self._world_comm.Send(
                        np.ascontiguousarray(gathered, dtype=np.complex128),
                        dest=dr,
                        tag=0,
                    )
            elif self._world_rank in secondary:
                recvbuf = np.empty(N_global, dtype=np.complex128)
                self._world_comm.Recv(recvbuf, source=primary, tag=0)
                ps_cache[i] = recvbuf

        # Phase 2: local pair loop, pure numpy.
        local_rhs_np = np.zeros(N_global, dtype=np.complex128)
        if self._worker_id >= 0 and self.my_ssm_quad_rhs_idc[j_idx]:
            y_buf = np.empty(N_global, dtype=np.complex128)
            for pair in self.my_ssm_quad_rhs_idc[j_idx]:
                idces = [
                    self._get_multiindex_index(self.ssm_multiindices, p)
                    for p in pair
                ]
                local_rhs_np += self.diff_eq.evaluate_quadratic_term(
                    0,
                    ps_cache[idces[0]],
                    ps_cache[idces[1]],
                    y_buf,
                )

        # Phase 3: Allreduce.
        global_rhs_np = np.empty_like(local_rhs_np)
        self._world_comm.Allreduce(
            local_rhs_np, global_rhs_np, op=MPI.SUM
        )

        # Phase 4: inject into the distributed rhs Vec.
        rhs.zeroEntries()
        r0, r1 = rhs.getOwnershipRange()
        rhs.setValues(
            np.arange(r0, r1, dtype=PETSc.IntType),
            global_rhs_np[r0:r1],
        )
        rhs.assemble()

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
        p0: Optional[PETSc.Vec] = None,
        verbose: int = 0,
    ) -> Tuple[List[PETSc.Vec], List[np.ndarray]]:
        r"""
        Compute the coefficients :math:`p_j` and :math:`g_j` in the
        polynomial expansions of :math:`P(s)` and :math:`g(s)`.

        This class assumes the reference ``c_*`` is a true equilibrium
        / periodic orbit of the host system, so the constant drift
        ``gs[0]`` is identically zero by construction.  For
        non-orbit references (e.g. atlas charts anchored at an
        intermediate point of a parent chart) use
        :class:`SpectralSubmanifoldChart` instead, which accepts a
        non-zero ``g0`` at instantiation and handles the resulting
        up-order coupling via Picard iteration.

        :param Phi: right eigenvectors
        :type Phi: SLEPc.BV
        :param Psi: left eigenvectors
        :type Psi: SLEPc.BV
        :param Lams: eigenvalues of shape :math:`(r,)`
        :type Lams: np.ndarray
        :param scaling: gauge scaling applied to the eigenvectors.  A scalar
            scales every master coordinate equally (V = Phi*scaling,
            W = Psi/scaling).  An array of shape (r,) applies a per-coordinate
            DIAGONAL gauge  V[:,i] = Phi[:,i]*scaling[i],
            W[:,i] = Psi[:,i]/scaling[i] -- useful to rebalance a strongly
            non-normal master subspace where one mode would otherwise dominate.
            Should be real (it must preserve W^H V = I).
        :type scaling: float or numpy.ndarray
        :param p0: optional zeroth-order manifold coefficient placed at
            ``ps[0]`` (the constant offset of the SSM from the origin in
            physical space).  ``None`` (default) keeps the zero vector.
        :type p0: Optional[PETSc.Vec]

        :return: manifold coefficients :math:`p_j` and
            dynamics coefficients :math:`g_j`
        :rtype: (List[PETSc.Vec], List[np.ndarray])
        """
        r = len(Lams)
        conj = self.conj_to_linear_dynamics

        # Normalise `scaling` to a per-coordinate gauge vector of shape (r,):
        # a scalar is broadcast to all master coordinates; an (r,) array gives
        # an independent (diagonal) gauge per coordinate.
        scaling = np.broadcast_to(
            np.asarray(scaling, dtype=PETSc.ScalarType), (r,)
        ).copy()

        # Scale eigenvectors per master coordinate:
        #   V[:, i] = Phi[:, i] * scaling[i],  W[:, i] = Psi[:, i] / scaling[i]
        # (a real diagonal gauge preserves W^H V = I; uniform scaling recovers
        # the original scalar behaviour).
        V = Phi.copy()
        W = Psi.copy()
        if np.all(scaling == scaling[0]):
            # Uniform gauge: scale the whole matrices in one shot (cheap).
            Vm = V.getMat(); Vm.scale(scaling[0]); V.restoreMat(Vm)
            Wm = W.getMat(); Wm.scale(1.0 / scaling[0]); W.restoreMat(Wm)
        else:
            for i in range(r):
                vc = V.getColumn(i); vc.scale(scaling[i]); V.restoreColumn(i, vc)
                wc = W.getColumn(i); wc.scale(1.0 / scaling[i]); W.restoreColumn(i, wc)

        _check_eigen_triplets(self.diff_eq, V, W, Lams)

        # Get a template vector for creating new PETSc Vecs
        v_ref = V.getColumn(0)
        template = v_ref.duplicate()
        V.restoreColumn(0, v_ref)

        # Zeroth and first order terms in the manifold expansion.
        # ps[0] = constant offset (user-supplied via `p0` or zero).
        # ps[k+1] = k-th master right eigenvector V[:, k].
        ps_zero = template.duplicate()
        if p0 is None:
            ps_zero.zeroEntries()
        else:
            p0.copy(ps_zero)            # in-place copy of user vector
        ps: List[PETSc.Vec] = [ps_zero]
        for k in range(r):
            v_k = V.getColumn(k)
            ps.append(v_k.copy())
            V.restoreColumn(k, v_k)

        # Zeroth and first order terms in g(s), internal dynamics.
        # gs[0]   = constant term (always zero — orbit reference).
        # gs[k+1] = linear term: Lams[k] in slot k, zero elsewhere — mirrors
        #          ps[k+1] = V[:, k] (k-th master eigenvector), giving a
        #          uniform polynomial representation g(s) = sum_j gs[j] s^j
        #          where the linear part reproduces  ds = Lams * s.
        gs: List[np.ndarray] = [
            np.zeros(r, dtype=complex) for _ in range(r + 1)
        ]
        for k in range(r):
            gs[k + 1][k] = Lams[k]

        # Reusable work vector
        rhs = template.duplicate()
        rhsj = template.duplicate()

        for j_idx in range(r + 1, len(self.ssm_multiindices)):
            j = self.ssm_multiindices[j_idx]
            if verbose == 1:
                from ..utils.miscellaneous import petscprint

                petscprint(
                    self.diff_eq.get_comm(),
                    f"Computing component {j} (order = {sum(j)})",
                )
            shift = PETSc.ScalarType(np.dot(Lams, np.asarray(j)))
            rhs.zeroEntries()

            # Compute contrib. from quadratic nature of the governing
            # equations.  Two paths:
            #   - n_workers == 1: original serial loop, bit-for-bit.
            #   - n_workers >  1: gather needed ps[i] to the workers
            #     that own them, run the pair-wise bilinear locally on
            #     COMM_SELF Vecs, then Allreduce the partial rhs back.
            if self.n_workers == 1:
                for pair in self.ssm_quad_rhs_idc[j_idx]:
                    idces = [
                        self._get_multiindex_index(self.ssm_multiindices, p)
                        for p in pair
                    ]
                    rhsj = self._evaluate_quadratic_rhs_root(
                        0, ps[idces[0]], ps[idces[1]], rhsj
                    )
                    rhs.axpy(1.0, rhsj)
            else:
                self._evaluate_quadratic_rhs_parallel(j_idx, ps, rhs)

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
                        rhs.axpy(
                            PETSc.ScalarType(-pair[0][k] * gs[idxg][k]),
                            ps[idxp],
                        )

            # Subtract V @ gj from rhs, then solve (shift*I - A) pj = rhs
            gj = W.dotVec(rhs) if not conj else np.zeros(r, dtype=complex)
            # Normal-form constraint: if this order is excluded from the
            # retained latent-space dynamics, force gj = 0 so the
            # corrections are absorbed into pj instead.
            if (
                self.latent_space_components is not None
                and sum(j) not in self.latent_space_components
            ):
                gj = np.zeros(r, dtype=complex)
            V.multVec(-1.0, 1.0, rhs, gj)  # rhs = -V @ gj + rhs
            pj = self.diff_eq.solve_linear_system(shift, rhs)

            if not conj:
                tol_conj = 1e-2
                # `scaling` is a per-coordinate vector; the element-wise product
                # recovers <Psi_i, pj> in the ORIGINAL (un-gauged) metric -- each
                # W_i carries a 1/scaling[i] that cancels -- so this acceptance
                # check is gauge-independent for scalar OR diagonal scaling.
                proj = np.asarray(W.dotVec(pj)).ravel() * scaling
                error_conj = np.linalg.norm(proj)
                if error_conj >= tol_conj:
                    from ..utils.miscellaneous import petscprint

                    petscprint(
                        self.diff_eq.get_comm(),
                        f"WARNING: |W^* pj| = {error_conj} > {tol_conj}. "
                        f"Consider modifying the scaling parameters when "
                        f"running .solve()",
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
            vec.axpy(PETSc.ScalarType(np.prod(s ** np.asarray(j))), self.ps[idx])
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

    def latent_space_dynamics(self, t: float, s: np.ndarray) -> np.ndarray:
        r"""
        Compute :math:`\dot{s} = \sum_j g_j \, s^j` summed over every
        multi-index.  This single uniform representation covers the
        constant term (``gs[0]``), the linear terms (``gs[1..r]``, which
        hold ``Lams`` diagonally to reproduce ``Lams * s``), and all
        nonlinear orders.  No special-casing per ``latent_space_components``
        is needed — orders the user excluded from the dynamics have
        ``gs[j] = 0`` exactly, so they contribute nothing.

        :param t: time (unused, included for ODE-solver compatibility)
        :type t: float
        :param s: latent-space coordinates of shape :math:`(r,)`
        :type s: np.ndarray

        :rtype: np.ndarray
        """
        if self.gs is None:
            raise RuntimeError("Call solve() first.")
        J = np.array(self.ssm_multiindices)
        G = np.array(self.gs)
        monomials = np.prod(s[None, :] ** J, axis=1)
        return monomials @ G

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
