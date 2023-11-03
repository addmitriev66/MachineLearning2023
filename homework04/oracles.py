import numpy as np
import scipy



class BaseSmoothOracle(object):
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, x):
        """
        Вычисляет значение функции в точке x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Вычисляет градиент в точке x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Вычисляет гессиан в точке x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Вычисляет phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Вычисляет phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        z = self.matvec_Ax(x)
        return np.mean(np.logaddexp(0, -self.b * z)) + 0.5 * self.regcoef * np.linalg.norm(x)**2

    def grad(self, x):
        z = self.matvec_Ax(x)
        sigmoid_term = expit(-self.b * z)
        gradient = -self.matvec_ATx(self.b * sigmoid_term) / len(self.b) + self.regcoef * x
        return gradient

    def hess(self, x):
        z = self.matvec_Ax(x)
        sigmoid_term = expit(-self.b * z)
        diag_sigmoid = sigmoid_term * (1 - sigmoid_term)
        hessian = self.matmat_ATsA(diag_sigmoid) / len(self.b) + self.regcoef * np.identity(len(x))
        return hessian


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        return np.mean(np.logaddexp(0, -self.b * (Ax + alpha * Ad))) + 0.5 * self.regcoef * np.linalg.norm(x + alpha * d) ** 2

    def grad_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        exp_term = -self.b * (Ax + alpha * Ad)
        coef = -self.b * np.exp(expit(exp_term))
        gradient = self.matvec_ATx(coef) + self.regcoef * (x + alpha * d)
        return gradient


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    if issparse(A):
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        matmat_ATsA = lambda s: (A.T * s).dot(A)
    else:
        matvec_Ax = lambda x: A @ x
        matvec_ATx = lambda x: A.T @ x
        matmat_ATsA = lambda s: A.T @ (s[:, None] * A)
    
    if oracle_type == 'usual':
        oracle = LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
    else:
        raise ValueError('Unknown oracle_type: {}'.format(oracle_type))
    
    return oracle


def grad_finite_diff(func, x, eps=1e-8):
    n = len(x)
    gradient = np.zeros(n)
    for i in range(n):
        x_plus_eps = x.copy()
        x_plus_eps[i] += eps
        gradient[i] = (func(x_plus_eps) - func(x)) / eps
    return gradient


def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))
