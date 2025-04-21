from .. import np
from .. import MPI
from .. import PETSc
from .. import SLEPc
from .. import typing

from ..utils.comms_helpers import compute_local_size
from ..utils.mat_helpers import create_dense_matrix


def compute_gramian_factors(
    L_generators: typing.Callable,
    frequencies: np.ndarray,
    weights: np.ndarray,
    B: SLEPc.BV,
    C: SLEPc.BV,
) -> typing.Tuple[PETSc.Mat, PETSc.Mat]:
    r"""
    Compute the Gramian factors as outlined in section 2 of [Dergham2011]_.
    In particular, we approximate the Reachability and Observability
    Gramians as follows

    .. math::

        \begin{align}
        G_R &= \frac{1}{2\pi}\int_{-\infty}^{\infty}R(\omega)BB^*R(\omega)^*\
            \,d\omega \approx \sum_j w_j X(\omega_j)X(\omega_j)^* = XX^* \\
        G_O &= \frac{1}{2\pi}\int_{-\infty}^{\infty}R(\omega)^*CC^*R(\omega)\,\
            d\omega \approx \sum_j w_j Y(\omega_j)Y(\omega_j)^* = YY^*
        \end{align}
        
    where :math:`\omega_j\in\Omega` are quadrature points and :math:`w_j` are
    corresponding quadrature weights. Here, :math:`X(\omega_j) = R(\omega_j)B`,
    and :math:`R(\omega) = \left(i\omega I - A\right)^{-1}`.
    When the linear operator :math:`A` is real valued, the Gramians may be 
    further manipulated into

    .. math::

        G_R = \frac{1}{\pi}\int_{0}^{\infty}R(\omega)BB^*R(\omega)^*\
        \,d\omega \approx \sum_j \delta_j w_j\left(X_\mathrm{r}(\omega_i)\
        X_\mathrm{r}(\omega)^* + X_{\mathrm{i}}(\omega_j)\
            X_{\mathrm{i}}(\omega_j)^*\right),

    where the subscripts "r" and "i" denote the real and imaginary parts,
    and :math:`\delta_j = 1` if :math:`\omega_j = 0` and 
    :math:`\delta_j = 1/2` otherwise.

    :param L_generators: tuple of functions that define the linear operator \
        :math:`R(\omega)^{-1}`, the action of :math:`R(\omega)` on matrices,
        and destructors to avoid memory leaks. (See examples.)
    :type L_generators: Tuple[[Callable, Callable, Tuple[Callable, Callable, \
        ...]]]
    :param frequencies: quadrature points :math:`\omega_j`
    :type frequencies: numpy.ndarray
    :param weights: quadrature weights :math:`w_j` (see description above to 
        see how the definition of :math:`w_j` changes depending on whether the 
        system is real-valued).
    :type weights: numpy.ndarray
    :param B: input matrix
    :type B: SLEPc.BV
    :param C: output matrix
    :type C: SLEPc.BV

    :return: tuple :math:`(X, Y)` (see definitions above)
    :rtype: Tuple[PETSc.Mat, PETSc.Mat]

    References
    ----------
    .. [Dergham2011] Dergham et al., *Model reduction for fluids using \
        frequential snapshots*, Physics of Fluids, 2011
    """
    nb = B.getSizes()[-1]
    nc = C.getSizes()[-1]
    idces = np.argsort(frequencies).reshape(-1)
    frequencies = frequencies[idces]
    weights = weights[idces]
    min_f = frequencies[0]
    if min_f >= 0.0:
        split = True
        nf = (
            2 * (len(frequencies) - 1) + 1
            if min_f == 0.0
            else 2 * len(frequencies)
        )
    else:
        split = False
        nf = len(frequencies)

    L, action, destroyers = L_generators[0](frequencies[0])
    action_ht = (
        L.apply_hermitian_transpose_mat
        if action == L.apply_mat
        else L.solve_hermitian_transpose_mat
    )

    LB = L.create_left_bv(nb)
    LC = L.create_right_bv(nc)
    Xarray = np.zeros((LB.getSizes()[0][0], nb * nf), dtype=np.complex128)
    Yarray = np.zeros((LC.getSizes()[0][0], nc * nf), dtype=np.complex128)
    for k in range(len(frequencies)):
        if k > 0:
            L, action, destroyers = L_generators[k](frequencies[k])
            action_ht = (
                L.apply_hermitian_transpose_mat
                if action == L.apply_mat
                else L.solve_hermitian_transpose_mat
            )
        LB = action(B, LB)
        LC = action_ht(C, LC)
        LB.scale(np.sqrt(weights[k]))
        LC.scale(np.sqrt(weights[k]))
        LBMat = LB.getMat()
        LBMat_ = LBMat.getDenseArray().copy()
        LB.restoreMat(LBMat)
        LCMat = LC.getMat()
        LCMat_ = LCMat.getDenseArray().copy()
        LC.restoreMat(LCMat)
        if not split:  # The system is complex-valued
            Xarray[:, k * nb : (k + 1) * nb] = LBMat_
            Yarray[:, k * nc : (k + 1) * nc] = LCMat_
        else:  # The system is real-valued
            delta = 1.0
            if k == 0:
                # Handle the zero frequency separately if necessary
                delta = np.sqrt(1 / 2) if frequencies[k] == 0.0 else delta
                Xarray[:, 0:nb] = delta * LBMat_.real
                Yarray[:, 0:nc] = delta * LCMat_.real
            else:
                k0 = nb + 2 * (k - 1) * nb
                k1 = k0 + 2 * nb
                Xarray[:, k0:k1] = delta * np.concatenate(
                    (LBMat_.real, LBMat_.imag), -1
                )
                k0 = nc + 2 * (k - 1) * nc
                k1 = k0 + 2 * nc
                Yarray[:, k0:k1] = delta * np.concatenate(
                    (LCMat_.real, LCMat_.imag), -1
                )
        for destroy in destroyers:
            destroy()
    Xsizes = (LB.getSizes()[0], Xarray.shape[-1])
    Ysizes = (LC.getSizes()[0], Yarray.shape[-1])
    X = PETSc.Mat().createDense(Xsizes, None, Xarray, MPI.COMM_WORLD)
    Y = PETSc.Mat().createDense(Ysizes, None, Yarray, MPI.COMM_WORLD)
    LB.destroy()
    LC.destroy()
    return (X, Y)


def compute_balanced_projection(
    X: SLEPc.BV, Y: SLEPc.BV, r: int
) -> typing.Tuple[SLEPc.BV, SLEPc.BV, np.ndarray]:
    r"""
    Given the output :math:`(X, Y)` of :func:`.compute_gramian_factors`, compute
    :math:`\Phi` and :math:`\Psi` (each of dimension :math:`N\times r`).
    Given the singular value decomposition :math:`Y^*X = U\Sigma V^*`, these
    are given by

    .. math::

        \Phi = X V \Sigma^{-1/2},\quad \Psi = Y U \Sigma^{-1/2}.

    :param X: reachability Gramian factor
    :type X: PETSc.Mat
    :param Y: observability Gramian factor
    :type Y: PETSc. Mat
    :param r: number of columns of :math:`\Phi` and :math:`\Psi`
    :type r: int

    :return: tuple :math:`(\Phi, \Psi, \Sigma)`
    :rtype: Tuple[SLEPc.BV, SLEPc.BV, numpy.ndarray]
    """
    comm = MPI.COMM_WORLD
    # Compute product Y^*@X
    Y.hermitianTranspose()
    Z = Y.matMult(X)
    Y.hermitianTranspose()
    svd = SLEPc.SVD().create(MPI.COMM_WORLD)
    svd.setOperators(Z)
    svd.setProblemType(SLEPc.SVD.ProblemType.STANDARD)
    svd.setType(SLEPc.SVD.Type.CROSS)
    svd.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
    svd.setUp()
    svd.solve()
    # Extract singular triplets
    r = np.min([r, svd.getConverged()])
    rloc = compute_local_size(r)
    u = Z.createVecLeft()
    v = Z.createVecRight()
    U = create_dense_matrix(comm, (Y.getSizes()[-1], (rloc, r)))
    V = create_dense_matrix(comm, (X.getSizes()[-1], (rloc, r)))
    S = create_dense_matrix(comm, ((rloc, r), (rloc, r)))
    U_ = U.getDenseArray()
    V_ = V.getDenseArray()
    S_ = np.zeros((r, r), dtype=np.float64)
    for i in range(r):
        s = svd.getSingularTriplet(i, u, v)
        U_[:, i] = u.getArray().copy()
        V_[:, i] = v.getArray().copy()
        S_[i, i] = s
        S.setValue(i, i, 1.0 / np.sqrt(s), addv=False)
    S.assemble(None)
    u.destroy()
    v.destroy()
    svd.destroy()
    # Compute matrices Phi and Psi for projection
    VS = V.matMult(S, None)
    US = U.matMult(S, None)
    Phi_mat = X.matMult(VS, None)
    Psi_mat = Y.matMult(US, None)
    Phi_ = SLEPc.BV().createFromMat(Phi_mat)
    Phi_.setType("mat")
    Phi = Phi_.copy()
    Psi_ = SLEPc.BV().createFromMat(Psi_mat)
    Psi_.setType("mat")
    Psi = Psi_.copy()
    objs = [Phi_, Psi_, Phi_mat, Psi_mat, U, V, S, Z, VS, US]
    for obj in objs:
        obj.destroy()
    return (Phi, Psi, S_)
