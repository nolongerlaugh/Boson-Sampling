import numpy as np

# generate random unitary matrix
def haar_measure(n):
    """
    Генерирует случайную унитарную матрицу
        
    n: dim of matrix
        
    return: matrix (n, n)
    """
    z = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.lib.scimath.sqrt(2.0)
    q,r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q

# recursive function for generating combinations
def find(n, range_m, output_state, output_states, index):
    """
    Вспомогательная функция для генерации одного состояния с фиксированной суммой
    
    n: сумма
    range_m: список возможных вариантов в одной ячейке
    output_state: текущее состояние
    output_states: список всех состояний
    index: текущий индекс в состоянии
    
    """
    if (n == 0):
        output_states.append(tuple(output_state.copy()))
        return
    if (n < 0 or index == range_m[-1]):
        return
    for i in range_m:
        output_state[index] = i
        find(n - i, range_m, output_state.copy(), output_states, index + 1)
        
# main function for generating combinations
def gen_output_states(m, n):
    """
    Функция, генерирующая список всевозможных состояний с фиксированной суммой
    
    m: кол-во индексов у одного состояния
    n: сумма
    
    
    """
    output_states = []
    cnts = np.arange(0, m + 1).tolist()
    output_state = [0] * m
    find(n, cnts, output_state, output_states, 0)
    return output_states

def calculate_permanent(A):
    """
    Перманент матрицы
    
    A: матрица
    
    return: перманент матрицы
        
    """
    n = A.shape[0]
    sz_mask = (2 ** n)
    dp = np.zeros(sz_mask, dtype=np.complex128)
    dp[0] = 1
    for mask in range(1, sz_mask):
        k = 0
        for i in range(n):
            if ((mask >> i) & 1) > 0:
                k += 1
        for i in range(n):
            if ((mask >> i) & 1) > 0:
                dp[mask] += dp[mask ^ (1 << i)] * A[k - 1, i]
    return dp[sz_mask - 1]