import numpy as np
import scipy


class CGL:
    r"""
        Class for the linear complex Ginzburg-Landau equation

        .. math::

            \partial_t q = \left(-\nu \partial_x + \
                \gamma\partial_x^2 + \mu\right)q,\quad q(x,t)\in\mathbb{C},
        
        where :math:`\mu = (\mu_0 - c_u^2) + \mu_2 x^2/2`, 
        :math:`\nu = U + 2 i c_u` and :math:`\gamma =  1 + i c_d`.
        Upon spatial discretization using a fourth-order central finite
        difference scheme, this leads to a system of ODEs

        .. math::

            d_t q = Aq,
        
        where now :math:`q` is a complex-valued vector. 

        :param x: array of unfiformly-spaced nodes :math:`x_i`
        :type x: numpy.array
        :param nu: (see above)
        :type nu: complex
        :param gamma: (see above)
        :type gamma: complex
        :param mu0: (see above)
        :type mu0: float
        :param mu2: (see above)
        :type mu2: float
        :param sigma: (see above)
        :type sigma: float
        """

    def __init__(self, x, nu, gamma, mu0, mu2, sigma):
        self.x = x
        self.nx = len(x)
        self.dx = x[1] - x[0]

        self.nu = nu
        self.gamma = gamma
        self.mu0 = mu0
        self.mu2 = mu2
        self.U = self.nu.real
        self.cu = self.nu.imag / 2
        self.cd = self.gamma.imag
        self.mu = (self.mu0 - self.cu**2) + self.mu2 * (self.x**2) / 2
        self.sigma = sigma

        # Additional parameters (needed for computation of exact eigenvalues)
        self.Umax = self.U + 2 * self.cd * self.cu
        self.mut = self.Umax**2 / (4 * np.abs(self.gamma) ** 2)
        self.h = np.sqrt(-2 * self.mu2 * self.gamma)
        self.muc = (
            self.mut + np.abs(self.h) * np.cos(np.angle(self.gamma) / 2) / 2
        )

        self.assemble_first_derivative_operator()
        self.assemble_second_derivative_operator()
        self.muI = scipy.sparse.diags(self.mu, offsets=0, format="csc")
        self.A = -self.nu * self.D + self.gamma * self.DD + self.muI
        self.M = self.dx * scipy.sparse.identity(self.nx)

    def assemble_first_derivative_operator(self):
        f = 1.0 / (12 * self.dx)
        s = 1.0 / (2 * self.dx)

        rows, cols, data = [], [], []

        for i in range(2, self.nx - 2):
            rows.extend([i, i, i, i])
            cols.extend([i - 2, i - 1, i + 1, i + 2])
            data.extend([f, -8 * f, 8 * f, -f])

        i = 1
        rows.extend([i, i, i])
        cols.extend([i - 1, i + 1, i + 2])
        data.extend([-8 * f, 8 * f, -f])

        i = self.nx - 2
        rows.extend([i, i, i])
        cols.extend([i - 2, i - 1, i + 1])
        data.extend([f, -8 * f, 8 * f])

        i = 0
        rows.append(i)
        cols.append(i + 1)
        data.append(s)

        i = self.nx - 1
        rows.append(i)
        cols.append(i - 1)
        data.append(-s)

        self.D = scipy.sparse.csc_array(
            (data, (rows, cols)), shape=(self.nx, self.nx)
        )

    def assemble_second_derivative_operator(self):
        f = 1.0 / (12 * (self.dx**2))
        s = 1.0 / (self.dx**2)

        rows, cols, data = [], [], []

        for i in range(2, self.nx - 2):
            rows.extend([i, i, i, i, i])
            cols.extend([i - 2, i - 1, i, i + 1, i + 2])
            data.extend([-f, 16 * f, -30 * f, 16 * f, -f])

        i = 1
        rows.extend([i, i, i, i])
        cols.extend([i - 1, i, i + 1, i + 2])
        data.extend([16 * f, -30 * f, 16 * f, -f])

        i = self.nx - 2
        rows.extend([i, i, i, i])
        cols.extend([i - 2, i - 1, i, i + 1])
        data.extend([-f, 16 * f, -30 * f, 16 * f])

        i = 0
        rows.extend([i, i])
        cols.extend([i, i + 1])
        data.extend([-2 * s, s])

        i = self.nx - 1
        rows.extend([i, i])
        cols.extend([i - 1, i])
        data.extend([s, -2 * s])

        self.DD = scipy.sparse.csc_array(
            (data, (rows, cols)), shape=(self.nx, self.nx)
        )

    def assemble_input_operators(self, xas):
        if hasattr(xas, "__len__"):
            B = np.zeros((self.nx, len(xas)))
            for k in range(len(xas)):
                B[:, k] = np.exp(
                    -((self.x - xas[k]) ** 2) / (2 * self.sigma**2)
                )
        else:
            B = np.exp(-((self.x - xas) ** 2) / (2 * self.sigma**2)).reshape(
                -1, 1
            )
        return B

    def assemble_output_operators(self, xas):
        if hasattr(xas, "__len__"):
            C = np.zeros((self.nx, len(xas)))
            for k in range(len(xas)):
                C[:, k] = np.exp(
                    -((self.x - xas[k]) ** 2) / (2 * self.sigma**2)
                )
        else:
            C = np.exp(-((self.x - xas) ** 2) / (2 * self.sigma**2)).reshape(
                -1, 1
            )

        return self.M.T @ C

    def compute_exact_eigenvalues(self, n):
        return np.asarray(
            [
                self.mu0
                - self.cu**2
                - self.nu**2 / (4 * self.gamma)
                - (j + 1 / 2) * self.h
                for j in range(n)
            ]
        )
