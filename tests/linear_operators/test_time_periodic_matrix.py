import numpy as np
import resolvent4py as res4py
from .. import pytest_utils


# ── helpers ─────────────────────────────────────────────────────────────────


def _build_op(comm, Alst_np, freqs, time):
    r"""Wrap a list of numpy Fourier-coefficient matrices into a
    :class:`TimePeriodicMatrixLinearOperator` (each coefficient gets
    wrapped in a :class:`MatrixLinearOperator` first)."""
    Apetsc_lst = [
        pytest_utils.numpy_to_petsc(comm, Ak_np) for Ak_np in Alst_np
    ]
    linop_lst = [
        res4py.linear_operators.MatrixLinearOperator(Apetsc)
        for Apetsc in Apetsc_lst
    ]
    op = res4py.linear_operators.TimePeriodicMatrixLinearOperator(
        linop_lst, freqs, time
    )
    return op


def _numpy_A_at(Alst_np, freqs, t, is_real_A):
    r"""Reference :math:`A(t) = \sum_k A_k e^{i \omega_k t}` in numpy.
    For one-sided ``freqs`` (``is_real_A=True``) the conjugate-symmetric
    partners are reconstructed on the fly: :math:`A_{-k}= \bar{A}_k`."""
    n = Alst_np[0].shape[0]
    A = np.zeros((n, n), dtype=np.complex128)
    for Ak, omega_k in zip(Alst_np, freqs):
        A += Ak * np.exp(1j * omega_k * t)
        if is_real_A and omega_k > 0.0:
            A += np.conj(Ak) * np.exp(-1j * omega_k * t)
    return A


def _make_real_A_coeffs(N, seed):
    r"""One-sided coeffs for a real :math:`A(t)`: ``freqs = [0, 1.2, 2.4]``,
    ``A_0`` real, ``A_1`` / ``A_2`` complex."""
    rng = np.random.default_rng(seed)
    freqs = np.array([0.0, 1.2, 2.4])
    A0 = rng.standard_normal((N, N)).astype(np.complex128)  # real DC
    A1 = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A2 = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    return [A0, A1, A2], freqs


def _make_complex_A_coeffs(N, seed):
    r"""Two-sided coeffs for a complex :math:`A(t)`:
    ``freqs = [-1.2, 0, 1.2]``, every coefficient complex."""
    rng = np.random.default_rng(seed)
    freqs = np.array([-1.2, 0.0, 1.2])
    coeffs = [
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        for _ in range(3)
    ]
    return coeffs, freqs


# ── real A(t) — one-sided freqs ────────────────────────────────────────────


def test_real_A_vec_real_input(comm):
    r"""Real :math:`A(t)` with one-sided freqs, real input vector — uses
    the ``vec_real``-and-double fast path inside ``apply`` /
    ``apply_hermitian_transpose``."""
    N = 8
    t = 0.7
    Alst_np, freqs = _make_real_A_coeffs(N, seed=11)
    op = _build_op(comm, Alst_np, freqs, t)
    A_t = _numpy_A_at(Alst_np, freqs, t, is_real_A=True)

    x, xp = pytest_utils.generate_random_vector(comm, N, complex=False)
    y = op.create_left_vector()
    err = pytest_utils.compute_error_vector(comm, op.apply, x, y, A_t.dot, xp)
    err_h = pytest_utils.compute_error_vector(
        comm, op.apply_hermitian_transpose, x, y, A_t.conj().T.dot, xp
    )

    x.destroy()
    y.destroy()
    for Aop in op.Alst:
        Aop.A.destroy()
        Aop.destroy()
    op.destroy()
    assert err < 1e-10, f"apply (real x) error = {err:.2e}"
    assert err_h < 1e-10, (
        f"apply_hermitian_transpose (real x) error = {err_h:.2e}"
    )


def test_real_A_vec_complex_input(comm):
    r"""Real :math:`A(t)`, complex input vector — exercises the
    ``A_{-k} v = conj(A_k conj(v))`` fallback for complex inputs."""
    N = 8
    t = 1.3
    Alst_np, freqs = _make_real_A_coeffs(N, seed=23)
    op = _build_op(comm, Alst_np, freqs, t)
    A_t = _numpy_A_at(Alst_np, freqs, t, is_real_A=True)

    x, xp = pytest_utils.generate_random_vector(comm, N, complex=True)
    y = op.create_left_vector()
    err = pytest_utils.compute_error_vector(comm, op.apply, x, y, A_t.dot, xp)
    err_h = pytest_utils.compute_error_vector(
        comm, op.apply_hermitian_transpose, x, y, A_t.conj().T.dot, xp
    )

    x.destroy()
    y.destroy()
    for Aop in op.Alst:
        Aop.A.destroy()
        Aop.destroy()
    op.destroy()
    assert err < 1e-10, f"apply (complex x) error = {err:.2e}"
    assert err_h < 1e-10, (
        f"apply_hermitian_transpose (complex x) error = {err_h:.2e}"
    )


def test_real_A_bv_real_input(comm):
    r"""Real :math:`A(t)`, real input BV — fast path for ``apply_mat`` /
    ``apply_hermitian_transpose_mat``."""
    N = 8
    s = 5
    t = 0.5
    Alst_np, freqs = _make_real_A_coeffs(N, seed=41)
    op = _build_op(comm, Alst_np, freqs, t)
    A_t = _numpy_A_at(Alst_np, freqs, t, is_real_A=True)

    X, Xp = pytest_utils.generate_random_bv(comm, (N, s), complex=False)
    Y = op.create_left_bv(s)
    err = pytest_utils.compute_error_bv(
        comm, op.apply_mat, X, Y, A_t.dot, Xp
    )
    err_h = pytest_utils.compute_error_bv(
        comm, op.apply_hermitian_transpose_mat, X, Y, A_t.conj().T.dot, Xp
    )

    X.destroy()
    Y.destroy()
    for Aop in op.Alst:
        Aop.A.destroy()
        Aop.destroy()
    op.destroy()
    assert err < 1e-10, f"apply_mat (real X) error = {err:.2e}"
    assert err_h < 1e-10, (
        f"apply_hermitian_transpose_mat (real X) error = {err_h:.2e}"
    )


def test_real_A_bv_complex_input(comm):
    r"""Real :math:`A(t)`, complex input BV — exercises the BV complex
    fallback via ``bv_conj`` and the lazy ``bvleftcc`` / ``bvrightcc``
    work buffers."""
    N = 8
    s = 5
    t = 2.1
    Alst_np, freqs = _make_real_A_coeffs(N, seed=53)
    op = _build_op(comm, Alst_np, freqs, t)
    A_t = _numpy_A_at(Alst_np, freqs, t, is_real_A=True)

    X, Xp = pytest_utils.generate_random_bv(comm, (N, s), complex=True)
    Y = op.create_left_bv(s)
    err = pytest_utils.compute_error_bv(
        comm, op.apply_mat, X, Y, A_t.dot, Xp
    )
    err_h = pytest_utils.compute_error_bv(
        comm, op.apply_hermitian_transpose_mat, X, Y, A_t.conj().T.dot, Xp
    )

    X.destroy()
    Y.destroy()
    for Aop in op.Alst:
        Aop.A.destroy()
        Aop.destroy()
    op.destroy()
    assert err < 1e-10, f"apply_mat (complex X) error = {err:.2e}"
    assert err_h < 1e-10, (
        f"apply_hermitian_transpose_mat (complex X) error = {err_h:.2e}"
    )


# ── complex A(t) — two-sided freqs, no conjugate-symmetry trick ────────────


def test_complex_A_vec(comm):
    r"""Complex :math:`A(t)` with two-sided freqs — every ``apply*``
    takes the generic path (no ``_real_A`` shortcut)."""
    N = 8
    t = 0.9
    Alst_np, freqs = _make_complex_A_coeffs(N, seed=67)
    op = _build_op(comm, Alst_np, freqs, t)
    A_t = _numpy_A_at(Alst_np, freqs, t, is_real_A=False)

    x, xp = pytest_utils.generate_random_vector(comm, N, complex=True)
    y = op.create_left_vector()
    err = pytest_utils.compute_error_vector(comm, op.apply, x, y, A_t.dot, xp)
    err_h = pytest_utils.compute_error_vector(
        comm, op.apply_hermitian_transpose, x, y, A_t.conj().T.dot, xp
    )

    x.destroy()
    y.destroy()
    for Aop in op.Alst:
        Aop.A.destroy()
        Aop.destroy()
    op.destroy()
    assert err < 1e-10, f"apply (complex A) error = {err:.2e}"
    assert err_h < 1e-10, (
        f"apply_hermitian_transpose (complex A) error = {err_h:.2e}"
    )


def test_complex_A_bv(comm):
    r"""Complex :math:`A(t)` with two-sided freqs — BV variants."""
    N = 8
    s = 4
    t = 1.7
    Alst_np, freqs = _make_complex_A_coeffs(N, seed=89)
    op = _build_op(comm, Alst_np, freqs, t)
    A_t = _numpy_A_at(Alst_np, freqs, t, is_real_A=False)

    X, Xp = pytest_utils.generate_random_bv(comm, (N, s), complex=True)
    Y = op.create_left_bv(s)
    err = pytest_utils.compute_error_bv(
        comm, op.apply_mat, X, Y, A_t.dot, Xp
    )
    err_h = pytest_utils.compute_error_bv(
        comm, op.apply_hermitian_transpose_mat, X, Y, A_t.conj().T.dot, Xp
    )

    X.destroy()
    Y.destroy()
    for Aop in op.Alst:
        Aop.A.destroy()
        Aop.destroy()
    op.destroy()
    assert err < 1e-10, f"apply_mat (complex A) error = {err:.2e}"
    assert err_h < 1e-10, (
        f"apply_hermitian_transpose_mat (complex A) error = {err_h:.2e}"
    )


# ── set_evaluation_time ────────────────────────────────────────────────────


def test_set_evaluation_time(comm):
    r"""Changing ``self.time`` via :meth:`set_evaluation_time` updates
    the exponential weights used by subsequent ``apply`` calls."""
    N = 8
    Alst_np, freqs = _make_real_A_coeffs(N, seed=97)
    op = _build_op(comm, Alst_np, freqs, time=0.0)
    x, xp = pytest_utils.generate_random_vector(comm, N, complex=False)
    y = op.create_left_vector()

    errors = []
    for t in (0.3, 1.1, 2.2):
        op.set_evaluation_time(t)
        A_t = _numpy_A_at(Alst_np, freqs, t, is_real_A=True)
        errors.append(
            pytest_utils.compute_error_vector(
                comm, op.apply, x, y, A_t.dot, xp
            )
        )

    x.destroy()
    y.destroy()
    for Aop in op.Alst:
        Aop.A.destroy()
        Aop.destroy()
    op.destroy()
    assert max(errors) < 1e-10, f"max error across times = {max(errors):.2e}"


# ── adjoint identity: <v, L w> = <L* v, w> ─────────────────────────────────


def test_adjoint_identity(comm):
    r"""Verify the adjoint identity

    .. math::

        \langle v, L w\rangle \;=\; \langle L^* v, w\rangle,
        \qquad \langle a, b\rangle = a^H b,

    for every combination of:

    * ``L`` real (one-sided ``freqs``) and ``L`` complex (two-sided),
    * ``v`` real and ``v`` complex,
    * ``w`` real and ``w`` complex,

    i.e. all 8 cases.  This exercises both the ``_real_A`` fast path
    (real input) and the explicit ``(k, -k)`` pair construction
    (complex input) on both :meth:`apply` and
    :meth:`apply_hermitian_transpose`.
    """
    N = 8
    t = 0.7

    L_configs = [
        ("real_A",    _make_real_A_coeffs(N, seed=131)),
        ("complex_A", _make_complex_A_coeffs(N, seed=137)),
    ]

    errors = {}
    for L_label, (Alst_np, freqs) in L_configs:
        op = _build_op(comm, Alst_np, freqs, t)
        for v_complex in (False, True):
            for w_complex in (False, True):
                v, _ = pytest_utils.generate_random_vector(
                    comm, N, complex=v_complex
                )
                w, _ = pytest_utils.generate_random_vector(
                    comm, N, complex=w_complex
                )

                Lw = op.apply(w)
                Lhv = op.apply_hermitian_transpose(v)

                # PETSc convention: x.dot(y) = y^H · x.
                # So <v, Lw> = v^H · (Lw) = Lw.dot(v)
                # and <L^H v, w> = (L^H v)^H · w = w.dot(Lhv).
                lhs = Lw.dot(v)
                rhs = w.dot(Lhv)

                scale = max(abs(lhs), 1e-300)
                err = abs(lhs - rhs) / scale
                errors[(L_label, v_complex, w_complex)] = err

                v.destroy()
                w.destroy()
                Lw.destroy()
                Lhv.destroy()
        for Aop in op.Alst:
            Aop.A.destroy()
            Aop.destroy()
        op.destroy()

    max_err = max(errors.values())
    msg = "\n".join(
        f"  {key}: rel error = {val:.2e}" for key, val in errors.items()
    )
    assert max_err < 1e-14, (
        f"max adjoint-identity rel error = {max_err:.2e}\n{msg}"
    )
