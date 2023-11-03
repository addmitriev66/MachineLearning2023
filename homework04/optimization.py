import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
from scipy.optimize import minimize_scalar
import time
from scipy.linalg import cho_factor, cho_solve



class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__


    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self.method == 'Wolfe':
            # Попытайтесь использовать библиотечную функцию scalar_search_wolfe2
            # Если она не сходится, запустите процедуру дробления шага
            try:
                alpha = scalar_search_wolfe2(lambda alpha: oracle.func_directional(x_k, d_k, alpha),
                                              lambda alpha: oracle.grad_directional(x_k, d_k, alpha),
                                              c1=self.kwargs.get('c1', 1e-4), c2=self.kwargs.get('c2', 0.9)).alpha
            except (ValueError, RuntimeError):
                alpha = self.backtracking_line_search(oracle, x_k, d_k, previous_alpha)
        else:
            # Если метод не 'Wolfe', используйте процедуру дробления шага
            alpha = self.backtracking_line_search(oracle, x_k, d_k, previous_alpha)
        return alpha

    def backtracking_line_search(self, oracle, x_k, d_k, previous_alpha=None):
        # Начальное значение шага
        alpha = 1.0 if previous_alpha is None else previous_alpha

        # Параметры для процедуры дробления шага
        rho = self.kwargs.get('rho', 0.5)  # Коэффициент уменьшения шага (обычно 0.5)
        c = self.kwargs.get('c', 1e-4)  # Параметр условия Армихо

        while True:
            # Вычисляем значение функции и градиента для текущего шага
            f_val = oracle.func(x_k + alpha * d_k)
            grad_val = oracle.grad(x_k + alpha * d_k)

            # Проверяем условие Армихо
            if f_val <= oracle.func(x_k) + c * alpha * np.dot(grad_val, d_k):
                return alpha  # Успешно найден подходящий шаг

            # Уменьшаем шаг и продолжаем итерации
            alpha *= rho

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()



def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iter_num in range(max_iter):
        # Вычисление градиента в текущей точке
        gradient_xk = oracle.grad(x_k)

        # Проверка критерия останова (норма градиента)
        grad_norm = np.linalg.norm(gradient_xk)
        if grad_norm < tolerance:
            return x_k, 'success', history

        # Вычисление направления поиска (отрицательный градиент)
        search_direction = -gradient_xk

        # Выполнение линейного поиска для определения размера шага (alpha)
        alpha = line_search_tool.line_search(oracle, x_k, search_direction)

        # Обновление текущей точки
        x_k = x_k + alpha * search_direction

        # Обновление истории для отслеживания
        if trace:
            history['time'].append(datetime.now())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

    # Если достигнуто максимальное количество итераций
    return x_k, 'iterations_exceeded', history



def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        # Вычисление градиента и гессиана
        grad_k = oracle.grad(x_k)
        hess_k = oracle.hess(x_k)
        
        # Вычисление направление Ньютона
        newton_direction = -np.linalg.solve(hess_k, grad_k)
        
        # Поиск строки
        alpha_k, _ = line_search_tool.line_search(oracle, x_k, newton_direction)
        
        # Обновление итерации
        x_k += alpha_k * newton_direction
        
        # Вычисление значения функции и норму градиента
        func_value = oracle.func(x_k)
        grad_norm = np.linalg.norm(grad_k)
        
        # Сохранение значений 
        if trace:
            history['time'].append(time.time())
            history['func'].append(func_value)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        
        # Проверка критерий сходимости
        if grad_norm < tolerance:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history




