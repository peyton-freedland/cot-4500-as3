from __init__ import *

def euler(f, y_0, t_0, t_n, h):
    t = np.arange(t_0, t_n + h, h)
    y = np.zeros(len(t))
    y[0] = y_0
    for i in range(1, len(t)):
        y[i] = y[i - 1] + (h * f(t[i - 1], y[i - 1]))
    print("%0.5f" %y[i])
    return None

def f(t, y):
    return (t - (y ** 2))

def runge_kutta(t_0, t_1, n, y_0):
    h = (t_1 - t_0) / n
    t = t_0
    y = y_0
    for i in range(0, n):
        k1 = h * f(t, y)
        k2 = h * f(t + (0.5 * h), y + (0.5 * k1))
        k3 = h * f(t + (0.5 * h), y + (0.5 * k2))
        k4 = h * f(t + h, y + k3)
        y_n = y + ((1.0 / 6.0) * (k1 + (2 * k2) + (2 * k3) + k4))
        t += h
        y = y_n
    print("%0.5f" %y)
    return None

def print_matrix(A: int, n: int):
    for i in range(n):
        print(*A[i])

def perform_operation(A: int, n: int):
    i = 0
    j = 0
    k = 0
    c = 0
    flag = 0
    for i in range(n):
        if(A[i][i] == 0):
            c = 1
            while (((i + c) < n) and A[i + c][i] == 0):
                c += 1
            if ((i + c) == n):
                flag = 1
                break
            j = i
            for k in range(1 + n):
                temp = A[j][k]
                A[j][k] = A[j + c][k]
                A[j + c][k] = temp
        for j in range(n):
            if (i != j):
                p = A[j][i] / A[i][i]
                k = 0
                for k in range(n + 1):
                    A[j][k] -= (A[i][k] * p)
    return flag

def print_result(A: int, n: int, flag: int):
    if (flag == 2):
        print("Infinitely many solutions exist")
    elif (flag == 3):
        print("No solutions exist")
    else:
        print("[", end = " ")
        for i in range(n):
            print(str(math.trunc(A[i][n] / A[i][i])) + '.', end = " ")
        print("\b]")

def check_consistency(A: int, n: int, flag: int):
    flag = 3
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += A[i][j]
        if (sum == A[i][j]):
            flag = 2
    return flag

def L_U_decomp(A):
    size = A.shape[0]
    U = A.copy()
    L = np.eye(size, dtype = np.double)
    for i in range(n):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]
    return L, U

def is_diagonally_dominant(matrix):
    size = len(matrix)
    for i in range(0, size):
        sum = 0
        for j in range(0, size):
            sum += abs(matrix[i][j])
        sum -= abs(matrix[i][i])
        if (abs(matrix[i][i]) < sum):
            return False
    return True

def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

euler(f, 1, 0, 2, 0.2)
print()

runge_kutta(0, 2, 10, 1)
print()

A_3 = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]], dtype = np.double)
n = 3
flag = 0
flag = perform_operation(A_3, n)
if (flag == 1):
    flag = check_consistency(A_3, n, flag)

print_result(A_3, n, flag)
print()

A_4 = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype = 'float')
print("%0.5f" %(np.linalg.det(A_4)))
print()
L, U = L_U_decomp(A_4)
print(L)
print()
print(U)
print()

A_5 = [[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]]
print(is_diagonally_dominant(A_5))
print()

A_6 = [[2, 2, 1], [2, 3, 0], [1, 0, 2]]
print(is_positive_definite(A_6))