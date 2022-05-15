import torch
import numpy as np

import numpy as np
import torch


class MPS(object):
    def __init__(self, info):
        self.tt_cores = []
        self.info = info
        self.N = None
        self.m = None
        self.r = []
        self.phys_ind = []
        self.bad = 0

    def all_zeros_state(self, n, m):
        """
        Создает начальный тензор в Tensor-train формате
        
        n: количество мод
        m: количество фотонов
        
        """
        self.tt_cores = []
        self.r = [1]
        self.phys_ind = []
        for i in range(n):
            c = np.zeros(m + 1)
            if i < m:
                c[1] = 1
            else:
                c[0] = 1
            self.tt_cores.append(torch.reshape(torch.tensor(c, dtype=self.info.data_type, device=self.info.device),
                                               (1, m + 1, 1)))
            self.r.append(1)
            self.phys_ind.append(m + 1)
        self.N = n
        self.m = m

    def one_qubit_gate(self, u, n):
        """
        Применяет однокубитный гейт к тензорному поезду
        
        u: matrix (m+1)x(m+1)
        n: индекс ядра, к которому нужно применить гейт
        
        """
        core = self.tt_cores[n]
        self.tt_cores[n] = torch.transpose(torch.tensordot(u, core, dims=([1], [1])), 0, 1)

    def two_qubit_gate(self, u, n, max_rank=None, ort=False):
        """
        Применяет двухкубитный гейт к тензорному поезду
        
        u: tensor (m+1)x(m+1)x(m+1)x(m+1)
        n: индекс ядра, к которому нужно применить гейт
        max_rank = ограничение ранга в SVD разложении
        ort: нужна ли ортонормализация
        """
        if ort:
            self.sequence_qr(n)
        phi = torch.tensordot(self.tt_cores[n], self.tt_cores[n + 1], dims=([2], [0]))
        phi = torch.tensordot(u, phi, dims=([2, 3], [1, 2]))
        phi = torch.transpose(phi, 0, 2)
        phi = torch.transpose(phi, 1, 2)
        unfolding = phi.reshape([self.r[n] * self.phys_ind[n], self.r[n + 2] * self.phys_ind[n + 1]])
        compressive_left, compressive_right = MPS.tt_svd(unfolding, max_rank)
        self.r[n + 1] = compressive_left.size()[1]
        self.tt_cores[n] = torch.reshape(compressive_left, [self.r[n], self.phys_ind[n], self.r[n + 1]])
        self.tt_cores[n + 1] = torch.reshape(compressive_right, [self.r[n + 1], self.phys_ind[n + 1], self.r[n + 2]])
        self.normalization(n)

    def normalization(self, n):
        """
        Нормализация ядра
        
        n: индекс ядра
        """
        self.tt_cores[n] = self.tt_cores[n] / self.get_norm()

    def return_full_tensor(self):
        """
        Возвращает полный тензор
        """
        full_tensor = self.tt_cores[0]
        for i in tqdm(range(1, len(self.tt_cores), 1)):
            full_tensor = torch.tensordot(full_tensor, self.tt_cores[i], dims=([-1], [0]))
        full_tensor = full_tensor.reshape(self.phys_ind)
        return full_tensor

    @staticmethod
    def tt_svd(unfolding, max_rank=None):
        """
        SVD-разложение матрицы
        
        unfolding: матрица
        max_rank: обрезка рангов
        """
        u, s, v = torch.linalg.svd(unfolding, full_matrices=False)
        s = s * (1.0 + 0.0 * 1j)
        if max_rank is not None:
            u = u[:, 0:max_rank]
            s = s[0:max_rank]
            v = v[0:max_rank, :]
        compressive_left = torch.tensordot(u, torch.diag(s), dims=([1], [0]))
        compressive_right = v
        return compressive_left, compressive_right

    @staticmethod
    def tt_svd_left(unfolding, rank=None):
        u, s, v = torch.linalg.svd(unfolding, full_matrices=False)
        s = s * (1.0 + 0.0 * 1j)
        q = u
        r = torch.tensordot(torch.diag(s), v, dims=([1], [0]))
        if rank is not None:
            q = q[:, 0:rank]
            r = r[0:rank, :]
        compressive_left = q
        compressive_right = r
        return compressive_left, compressive_right

    @staticmethod
    def tt_svd_right(unfolding, rank=None):
        u, s, v = torch.linalg.svd(torch.transpose(torch.conj(unfolding), 0, 1), full_matrices=False)
        s = s * (1.0 + 0.0 * 1j)
        q = u
        r = torch.tensordot(torch.diag(s), v, dims=([1], [0]))
        l = torch.transpose(torch.conj(r), 0, 1)
        q = torch.transpose(torch.conj(q), 0, 1)
        if rank is not None:
            l = l[:, 0:rank]
            q = q[0:rank, :]
        compressive_left = l
        compressive_right = q
        return compressive_left, compressive_right

    def sequence_qr_left(self, n):
        for i in range(0, n, 1):
            phi = torch.tensordot(self.tt_cores[i], self.tt_cores[i + 1], dims=([2], [0]))
            unfolding = torch.reshape(phi, (self.r[i] * self.phys_ind[i], self.phys_ind[i + 1] * self.r[i + 2]))
            # compressive_left, compressive_right = MPS.tt_qr(unfolding, rank=self.r[i + 1])
            compressive_left, compressive_right = MPS.tt_svd_left(unfolding, rank=self.r[i + 1])
            self.tt_cores[i] = torch.reshape(compressive_left, (self.r[i], self.phys_ind[i], self.r[i + 1]))
            self.tt_cores[i + 1] = torch.reshape(compressive_right, (self.r[i + 1], self.phys_ind[i + 1],
                                                                     self.r[i + 2]))

    def sequence_qr_right(self, n):
        for i in range(self.N - 1, n + 1, -1):
            phi = torch.tensordot(self.tt_cores[i - 1], self.tt_cores[i], dims=([2], [0]))
            unfolding = torch.reshape(phi, (self.r[i - 1] * self.phys_ind[i - 1], self.phys_ind[i] * self.r[i + 1]))
            # compressive_left, compressive_right = MPS.tt_lq(unfolding, rank=self.r[i])
            compressive_left, compressive_right = MPS.tt_svd_right(unfolding, rank=self.r[i])
            self.tt_cores[i - 1] = torch.reshape(compressive_left, (self.r[i - 1], self.phys_ind[i - 1], self.r[i]))
            self.tt_cores[i] = torch.reshape(compressive_right, (self.r[i], self.phys_ind[i], self.r[i + 1]))

    def sequence_qr(self, n):
        """
        qr-разложение
        n - индекс ядра
        """
        if n == 0:
            self.sequence_qr_right(n)
            pass
        elif n == (self.N - 2):
            self.sequence_qr_left(n)
        else:
            self.sequence_qr_left(n)
            self.sequence_qr_right(n)

    def get_norm(self):
        """
        Норма тензора
        """
        core_prev = torch.tensordot(self.tt_cores[0], torch.conj(self.tt_cores[0]), dims=([1], [1]))
        for i in range(1, len(self.tt_cores), 1):
            core_prev = torch.tensordot(core_prev, self.tt_cores[i], dims=([1], [0]))
            core_prev = torch.tensordot(core_prev, torch.conj(self.tt_cores[i]), dims=([2], [0]))
            core_prev = torch.einsum('ijklkn', core_prev)
            core_prev = torch.transpose(core_prev, 1, 2)
        norm_square = core_prev[0][0][0][0]
        norm = torch.abs(torch.sqrt(norm_square))
        return norm

    def get_element(self, list_of_index):
        """
        Возвращает элемент тензора по индексу
        
        list_of_index: индекс
        """
        matrix_list = [self.tt_cores[i][:, index, :] for i, index in enumerate(list_of_index)]
        element = matrix_list[0]
        for matrix in matrix_list[1:]:
            element = torch.tensordot(element, matrix, dims=([1], [0]))
        return element[0][0]
    
    def prepare_gen(self, seed=239):
        """
        предварительная настройка перед генерацией
        
        seed: random seed
        """
        np.random.seed(seed)
        self.core_prev_list = [torch.tensordot(self.tt_cores[0], torch.conj(self.tt_cores[0]), dims=([1], [1]))]
        for i in range(1, len(self.tt_cores), 1):
            core_prev = self.core_prev_list[-1]
            core_prev = torch.tensordot(core_prev, self.tt_cores[i], dims=([1], [0]))
            core_prev = torch.tensordot(core_prev, torch.conj(self.tt_cores[i]), dims=([2], [0]))
            core_prev = torch.einsum('ijklkn', core_prev)
            core_prev = torch.transpose(core_prev, 1, 2)
            self.core_prev_list.append(core_prev)
    
    def gen_state(self):
        """
        сгенерировать одно состояние
        """
        state = []
        sm = 0
        vec = torch.ones(1, dtype=torch.complex128)
        for i in range(len(self.tt_cores) - 1):
            core = self.tt_cores[-i-1]
            core = torch.tensordot(core, vec, dims=([2], [0]))[:, :, None]
            core_prev = self.core_prev_list[-i-2]
            core_prev = torch.tensordot(core_prev, core, dims=([1], [0]))
            core_prev = torch.tensordot(core_prev, torch.conj(core), dims=([2], [0]))
            res = torch.sum(core_prev, dim=[0, 1, 2, 3, 5])
            res = torch.abs(res)
            res /= torch.sum(res)
            p = np.random.choice(self.m + 1, p=res.numpy())
            vec = (self.tt_cores[-i-1][:, p, :] @ vec)
            state.append(p)
            sm += p
            if sm > self.m:
                return state
        state.append(self.m - sm)
        return list(reversed(state))
    
    def gen_state_great(self):
        """
        сгенерировать одно состояние, но пропуская "плохие состояния"
        """
        while True:
            state = self.gen_state()
            if sum(state) == self.m:
                return tuple(state)
            else:
                self.bad += 1
                

                
import torch
from lib import haar_measure


class OneCubit:
    # u -> one number
    # i -> index
    def __init__(self, u, i):
        self.u = u
        self.i = i
        
    def get_matrix(self, n):
        """
        Возвращает матрицу из разложения матрицы интерферометра
        
        n: кол-во бозонов
        """
        T = np.diag(np.ones(n, dtype=np.complex128))
        T[self.i, self.i] = self.u
        return T

    def get_gate(self, m):
        """
        Возвращает специально построенную матрицу двухкубитного гейта
        
        m: кол-во мод
        """
        diag = np.ones(m + 1, dtype=np.complex128)
        for i in range(1, m + 1):
            diag[i] = diag[i - 1] * self.u
        return torch.diag(torch.tensor(diag, dtype=torch.complex128))


class TwoCubit:
    # u -> 2x2 matrix
    # i -> index
    def __init__(self, u, i):
        self.u = u
        self.i = i
    
    def get_matrix(self, n):
        """
        Возвращает матрицу из разложения матрицы интерферометра
        
        n: кол-во бозонов
        """
        T = np.diag(np.ones(n, dtype=np.complex128))
        T[self.i, self.i] = self.u[0, 0]
        T[self.i + 1, self.i] = self.u[1, 0]
        T[self.i, self.i + 1] = self.u[0, 1]
        T[self.i + 1, self.i + 1] = self.u[1, 1]
        return T
    
    def get_gate(self, m):
        """
        Возвращает специально построенную матрицу двухкубитного гейта
        
        m: кол-во мод
        """
        factlog = np.zeros(m + 1)
        for i in range(1, m + 1):
            factlog[i] = factlog[i - 1] + np.log(i)
        def getClog(n, k):
            return factlog[n] - factlog[k] - factlog[n - k]
        u_power = [np.ones((2, 2), dtype=np.complex128)]
        for i in range(1, m + 1):
            u_power.append(u_power[-1] * self.u.T)
        
        result = torch.zeros((m + 1, m + 1, m + 1, m + 1), dtype=torch.complex128)
        for n1 in range(m + 1):
            for n2 in range(m - n1 + 1):
                for k1 in range(n1 + 1):
                    for k2 in range(n2 + 1):
                        rlog = 0.0
                        rlog -= factlog[n1] / 2
                        rlog -= factlog[n2] / 2
                        rlog += getClog(n1, k1)
                        rlog += getClog(n2, k2)
                        rlog += factlog[n1 + n2 - k1 - k2] / 2
                        rlog += factlog[k1 + k2] / 2
                        coef = np.exp(rlog)
                        coef *= u_power[n1 - k1][0, 0]
                        coef *= u_power[k1][1, 0]
                        coef *= u_power[n2 - k2][0, 1]
                        coef *= u_power[k2][1, 1]
                        result[n1, n2, n1 + n2 - k1 - k2, k1 + k2] += coef
        
        return result


def composition(n, operations):
    """
    По всем операциям строит матрицу интерферометры
    
    n: размер матрицы
    operations: список из всех операций
    """
    A = np.diag(np.ones(n, dtype=np.complex128))
    for operation in operations:
        A = (A @ operation.get_matrix(n))
    return A


EPS = 1e-6


def decomposition(A):
    """
    Матрицу интерферометра расскладывает по операциям преобразований
    
    A: матрица интерферометра
    """
    A = A.copy()
    
    left_operations = []
    right_operations = []
    
    n = A.shape[0]
    for i in range(n - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                ix = n - j - 1
                iy = i - j
                x0 = A[ix, iy]
                y0 = A[ix, iy + 1]
                if np.abs(x0) < EPS:
                    continue
                r = (y0 / x0)
                ctg = abs(r)
                r /= ctg
                v = np.sqrt(ctg * ctg + 1)
                sin = 1 / v
                cos = ctg / v
                
                T = np.diag(np.ones(n, dtype=np.complex128))
                T[iy, iy] = r * cos
                T[iy, iy + 1] = r * sin
                T[iy + 1, iy] = -sin
                T[iy + 1, iy + 1] = cos
                A = (A @ T)
                
                r = r.conj()
                u = np.zeros((2, 2), dtype=np.complex128)
                u[0, 0] = r * cos
                u[1, 0] = r * sin
                u[0, 1] = -sin
                u[1, 1] = cos
                left_operations.append(TwoCubit(u, iy))
        else:
            for j in range(i + 1):
                ix = n + j - i - 1
                iy = j
                x0 = A[ix, iy]
                y0 = A[ix - 1, iy]
                if np.abs(x0) < EPS:
                    continue
                r = -(y0 / x0)
                ctg = abs(r)
                r /= ctg
                r = r.conj()
                v = np.sqrt(ctg * ctg + 1)
                sin = 1 / v
                cos = ctg / v
                
                T = np.diag(np.ones(n, dtype=np.complex128))
                T[ix - 1, ix - 1] = r * cos
                T[ix, ix - 1] = r * sin
                T[ix - 1, ix] = -sin
                T[ix, ix] = cos
                A = (T @ A)
                
                r = r.conj()
                u = np.zeros((2, 2), dtype=np.complex128)
                u[0, 0] = r * cos
                u[1, 0] = -sin
                u[0, 1] = r * sin
                u[1, 1] = cos
                right_operations.append(TwoCubit(u, ix - 1))
    
    left_operations = list(reversed(left_operations))
    operations = right_operations + [OneCubit(A[i, i], i) for i in range(n)] + left_operations
    return operations

class Info(object):
    def __init__(self, data_type=torch.complex128, device="cpu"):
        self.data_type = data_type
        self.device = device
        
        
from lib import *
import bisect as bs
import time
from tqdm import tqdm

class Sampler():
    
    def __init__(self, m, n, U=None):
        assert (m >= n)
        self.m = m
        self.n = n
        if U is None:
            self.U = haar_measure(m)
        else:
            self.U = U
        self.x = []
        self.x_index = {}
        self.A = np.abs((self.U.T[:n]).T)**2
        self.input_state = tuple([1]*n + [0]*(m -n))
        self.ps = {}
        self.pds = {}
    
    
    def eval_all_x(self):
        """
        Генерирует всевозможные состояния с фиксированной суммой
        """
        self.x = gen_output_states(self.m, self.n)

        
    def As_matrix(self, A, S):
        """
        Строит матрицу по выходному состоянию
        
        A: матрица интерферометра
        S: состояние
        """
        res = np.arange(0, 0)
        for i, s in enumerate(S, 0):
            for j in range(0, s):
                if (res.size == 0):
                    res = np.array(A[i])
                else:
                    res = np.row_stack((res, A[i]))
        return res

    
    def pr_pd(self, output_state):
        """
        Вычисляет вероятность состояния как различимых частиц
        
        output_state: состояние
        """
        if output_state in self.pds:
            v = self.pds[output_state]
        else:
            v = -1
        if (v == -1):
            res = calculate_permanent(self.As_matrix(self.A, output_state))
            for s in output_state:
                res /= np.math.factorial(s)
            self.pds[output_state] = res
            return res
        else:
            return v

        
    def pr(self, output_state):
        """
        Вычисляет вероятность состояния как неразличимых частиц
        
        output_state: состояние
        """
        if output_state in self.ps:
            v = self.ps[output_state]
        else:
            v = -1
        if ( v == -1):
            UST = self.As_matrix(self.U.T[:self.n].T, output_state)
            res = np.abs(calculate_permanent(UST)) ** 2
            for s in output_state:
                res /= np.math.factorial(s)
            self.ps[output_state] = res
            #self.density[output_state] = res
            return res
        else:
            return v

        
    def sample_from_density(self, a, k=1):
        """
        Генерирует точку из известного распределения
        
        a: распределение
        k: количество точек
        
        return: выходное состояние
        """
        b = np.cumsum(a)
        if (k == 1):
            i = np.random.random()
            return (bs.bisect_left(b, i))
        res = []
        for i in range(0, k):
            res.append(self.x[bs.bisect_left(b, np.random.random())])
        return (res)

    
    def gen_from_pr(self):
        """
        генерирует точку из различимого распределения
        
        return: состояние
        """
        h = []
        s = []
        for j in range(0, self.n):
            a = self.A.T[j]
            h.append(self.sample_from_density(a))
        for i in range(0, self.m):
            s.append(h.count(i))
        return (tuple(s))

    
    def gen_one_point_mis(self, x_t, pr_x_t, pr_pd_x_t):
        """
        генерирует точку из неразличимого распределения
        
        x_t: предыдущая точка
        pr_x_t: вероятность этой точки как из распределения неразличимых частиц
        pr_pd_x_t: вероятность этой точки как из распределения различимых частиц
        
        return:
        x_: сгенерированная точка
        pr_x_: вероятность этой точки как из распределения неразличимых частиц
        pr_pd_x_: вероятность этой точки как из распределения различимых частиц
        """
        x_ = self.gen_from_pr()
        pr_x_ = self.pr(x_)
        pr_pd_x_ = self.pr_pd(x_)
        a1 = pr_x_ / pr_x_t
        a2 = pr_pd_x_t / pr_pd_x_
        a = a1 * a2
        if (a >= 1):
            return (x_, pr_x_, pr_pd_x_)
        ran = np.random.random()
        if (ran > a):
            return (x_t, pr_x_t, pr_pd_x_t)
        return (x_, pr_x_, pr_pd_x_)

    
    def sample_mis(self, k, t1=1, t2=1):
        """
        Генерирует точку из неразличимого распределения с помощью МСМС
        
        k: количество точек
        t1: префикс
        t2: шаг
        """
        val = []
        x_t = self.gen_from_pr()
        val.append(x_t)
        val.append(self.pr(x_t))
        val.append(self.pr_pd(x_t))
        res = []
        for i in range(0, t1):
            val = self.gen_one_point_mis(val[0], val[1], val[2])
        l = 0
        res.append(tuple(val[0]))
        while (len(res) != k):
            l += 1
            val = self.gen_one_point_mis(val[0], val[1], val[2])
            if (l == t2):
                res.append(tuple(val[0]))
                l = 0
        return (res)

    
    def eval_density(self):
        """
        вычисляет полное распределение
        
        return: полное распределение (индексы соответствуют индексам self.x)
        """
        self.eval_all_x()
        self.density = [0] * len(self.x)
        for i, s in enumerate(self.x):
            self.density[i] = self.pr(s)
        return self.density

            
    def check_sample_coef(self, d):
        """
        Рассчитать Fidelity для выборки
        
        d: выборка
        """
        uniq = {}
        ld = len(d)
        sum_l = 0
        for s in d:
            if s in uniq:
                uniq[s] += 1
            else:
                uniq[s] = 1
        for items in uniq.items():
            sum_l += (self.pr(items[0])*items[1]/ld)**0.5
        return sum_l

    
    def delete_ps(self):
        """
        Чистим сохраненные состояния
        """
        self.ps = {}
        self.pds = {}

    def do_MPS_prepare(self, max_rank=None, ort_period = 0):
        """
        Предварительная настройка для генерации с помощью MPS
        
        max_rank: максимальный ранг тензора
        ort_period: период, с которым нужно делать ортонормализацию
        """
        A = self.U
        n = self.m
        m = self.n
        operations = decomposition(A)
        mps = MPS(Info())
        mps.all_zeros_state(n, m)
        roperations = list(reversed(operations))
        l = len(roperations)
        period = 0
        i = 0
        for operation in tqdm(roperations):
            if isinstance(operation, TwoCubit):
                if (period == ort_period or i == l - 1):
                    mps.two_qubit_gate(operation.get_gate(m), operation.i, max_rank, True)
                    period = 0
                else:
                    mps.two_qubit_gate(operation.get_gate(m), operation.i, max_rank, False)
                    period += 1
            else:
                mps.one_qubit_gate(operation.get_gate(m), operation.i)
            i += 1
        mps.prepare_gen()
        self.mps = mps
    
    def sample_mps(self, k):
        """
        Генерация с помощью MPS
        
        k: кол-во точек
        """
        states = []
        for i in tqdm(range(k)):
            states.append(self.mps.gen_state_great())
        return states
        
    def average_score_value(self, t2, k, num_tests=100, t1=100):
        """
        Среднее Fidelity для MCMC
        t2: шаг
        k: кол-во точек
        num_tests: кол-во тестов
        t1: префикс
        
        return:
                среднее фиделити
                время генерации num_tests сэмплов
        """
        self.delete_ps()
        sum_score = 0
        time_to_sample = 0
        for test in range(0, num_tests):
            start_time = time.time()
            d = self.sample_mis(k, t1, t2)
            time_to_sample += time.time() - start_time
            sum_score += self.check_sample_coef(d)
        return (sum_score/ num_tests, time_to_sample)

    
    def get_k_value(self, t2, k_start=1, k_step = 50, num_tests=100, t1=100):
        """
        Находит минимальное количество точек k в выборке, при котором она совпадает с распределением на 0.95 
        
        t2 - шаг MCMC
        k_start - начальное k
        k_step - шаг k
        num_tests - количество тестов
        t1 - префикс MCMC
        """
        k = k_start
        while (k <= k_start*10000):
            avg_score = self.average_score_value(t2, k, num_tests, t1)
            if (avg_score[0] >= 0.95):
                return (k, avg_score[1])
            k += k_step
            print(f'\r{k}', end='')
        return (-1, 0.0)
    
    def ratio_test(self, d):
        """
        Тест отношения правдоподобия
        
        d - выборка
        
        Return: p_ind выборки
        """
        r = []
        q = []
        for x in d:
            r.append(self.pr_pd(x))
            q.append(self.pr(x))
        r = np.array(r, dtype=np.float64)
        q = np.array(q, dtype=np.float64)
        hi = np.prod(q/r)
        return hi/(1 + hi)