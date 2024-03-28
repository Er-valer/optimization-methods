import math
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import numpy.linalg as la

import scipy.optimize as opt

INTERVAL = (-20, 20)
COLORMAP_FILL = ListedColormap(plt.get_cmap("bone_r")(np.linspace(0, 0.5, 256)))
COLORMAP_CONTOUR = ListedColormap(plt.get_cmap("bone_r")(np.linspace(0.25, 0.75, 256)))
PHI = (math.sqrt(5) + 1) / 2

def printPlots2D(f, interval, trace, delta = 1e-5):
    x = np.arange(interval[0], interval[1], delta)
    X = x

    Y = f(X)

    plt.plot(X, Y)
    plt.plot(*zip(*trace))

    plt.show()

def printPlots3D(f, optimum, trace, delta=1e-2, dots=1e3):
    xOpt, yOpt = optimum
    xMax = max(max(map(lambda args: abs(args[0] - xOpt), trace)), delta)*1.1
    yMax = max(max(map(lambda args: abs(args[1] - yOpt), trace)), delta)*1.1
    optMax = max(xMax, yMax)
    intervalX = max(-1e9, -optMax + xOpt), min(1e9, optMax + xOpt)
    intervalY = max(-1e9, -optMax + yOpt), min(1e9, optMax + yOpt)

    x = np.arange(intervalX[0], intervalX[1], (intervalX[1]-intervalX[0])/dots)
    y = np.arange(intervalY[0], intervalY[1], (intervalY[1]-intervalY[0])/dots)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)

    contour = plt.contour(X, Y, Z, levels=15, cmap=COLORMAP_CONTOUR)
    plt.contourf(X, Y, Z, levels=15, cmap=COLORMAP_FILL)
    plt.clabel(contour)
    plt.plot(*zip(*trace), color="red", marker='.')

    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm")

    plt.show()


# Для евклидовой нормы число обусловленности - отношение наибольшего сингулярного числа к наименьшему
# Функция возращает матрицу с заданным отношением, построенную с помощью SVD (singular value decomposition)
def matrix_by_condition(n, cond):
    U, _ = la.qr(np.random.rand(n, n))

    diag = np.random.randint(1, cond + 1, size=n)
    diag[0] = 1
    diag[-1] = cond
    S = np.diag(diag)

    ans = U @ S @ U.T # Избегаем гиперболических параболоидов
    return ans

def function_by(dim, cond):
    A = matrix_by_condition(dim, cond)
    def f(*args):
        x = np.array(args)
        return x @ A @ x.T

    def grad(*args):
        x = np.array(args)
        return 2 * A @ x.T
    
    def printFunc(*args):
        x = np.array(args)
        return np.einsum('i..., ij, j...', x, A, x)

    return f, grad, printFunc

def add_noise(f, grad, prF, noise):
    def f_noise(*args):
        return f(*args) + (random.random() - 0.5)*noise

    def grad_noise(*args):
        return grad(*args) + (random.random() - 0.5)*noise
    
    def prF_noise(*args):
        prF_clear = prF(*args)
        return  prF_clear + (np.random.rand(*prF_clear.shape) - 0.5)*noise
    
    return f_noise, grad_noise, prF_noise

def dichotomy(f, interval, N = 100, eps=1e-3):
    count = 0
    l, r = interval
    m = (l + r)/2
    fm = f(m)

    trace = [(m, fm)]

    for _ in range(N):
        trace.append((m, fm))
        x1, x2 = (l + m)/2, (m + r)/2
        fx1 = f(x1)

        if (fx1 < fm):
            l, m, r = l, x1, m
            fm = fx1
            count += 1
        else:
            fx2 = f(x2)
            if (fm < fx2):
                l, m, r = x1, m, x2
            else:
                l, m, r = m, x2, r
                fm = fx2
            count += 2
        
        if ((r - l)/2 < eps):
            break
    
    return m, trace, count

def golden_ratio(f, interval, N = 100, eps=1e-3):
    count = 0
    l, r = interval
    x1, x2 = r - (r - l)/PHI, l + (r - l)/PHI
    fx1, fx2 = f(x1), f(x2)

    trace = []

    for _ in range(N):
        if (fx1 < fx2):
            r, x2 = x2, x1
            fx2 = fx1

            x1 = r - (r - l)/PHI
            fx1 = f(x1)
        else:
            l, x1 = x1, x2
            fx1 = fx2

            x2 = l + (r - l)/PHI
            fx2 = f(x2)

        count += 1
        if ((r - l)/2 < eps):
            break
    
    return (l + r)/2, trace, count

def find_interval(f, step = 1e-1):
    s = step
    while f(s) < f(0):
        s += step

    return (0, s)

def find_best_step(f, x, direction, method):
    g = lambda s: f(*(x + s * direction))
    return method(g, find_interval(g))

def gradient_descent(f, grad, start, method, step=None, N=1000000, eps=1e-3):
    countF = 0
    countGrad = 0
    countIt = 0

    x = start
    x_pr = None

    trace = [x]

    for _ in range(N):
        countIt += 1
        direction = -grad(*x)
        countGrad += 1

        if step:
            lr = step
        else:
            best_step = find_best_step(f, x, direction, method)
            countF += best_step[2]
            lr = best_step[0]
        
        x_pr = x
        x = x + lr * direction

        trace.append(x)

        if (la.norm(x - x_pr) <= eps):
            break

    return x, trace, countIt, countF + countGrad

def nelder_mead(f, start, N = 100000, eps = 1e-3):
    f_ = lambda x: f(*x)
    result = opt.minimize(f_, start, method='Nelder-Mead', options={'return_all': True, 'maxiter': N, 'xatol': eps})
    return result["x"], result["allvecs"], result["nit"], result["nfev"]


fs = [lambda x, y: x**2 + y**2 - x*y,
    lambda x, y: x**4 + y**2 + (x**2)*y,
    lambda x, y: (1 - x)**2 + 100*(y - x**2)**2]

realOpts = [np.array([0, 0]), np.array([0, 0]), np.array([1, 1])]

grads = [lambda x, y: np.array([2*x - y, 2*y - x]),
    lambda x, y: np.array([2*x*(2*x**2 + y), x**2 + 2*y]),
    lambda x, y: np.array([2*(-1 + x + 200*x**3 - 200*x*y),  200*(-x**2 + y)])]

bestConstStep = [1/3, 0.526209/2, 0.0015]

startsSimple = [np.array([1, 1])]
startsOnCircle = [np.array([1, 0]), np.array([math.sqrt(2)/2, math.sqrt(2)/2]), np.array([0, 1]), np.array([-math.sqrt(2)/2, math.sqrt(2)/2]), np.array([-1, 0]), np.array([-math.sqrt(2)/2, -math.sqrt(2)/2]), np.array([0, -1]), np.array([math.sqrt(2)/2, -math.sqrt(2)/2])]
startsOnLine = [np.array([1/64, 0]), np.array([1/32, 0]), np.array([1/16, 0]), np.array([1/8, 0]), np.array([1/4, 0]), np.array([1/2, 0]), np.array([1, 0])]

for i in range(2, len(fs)):
    print(f"------ f = {i}")
    for start_ in startsSimple:
        #print(f"--------------- start = {start}")
        for eps in [1e-7]:#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            start = start_#/16 + realOpts[i]
            #optimum, trace, countIt, countF = nelder_mead(fs[i], start, eps=eps)
            #optimum, trace, countIt, countF = gradient_descent(fs[i], grads[i], start, None, step=bestConstStep[i], eps=eps)
            optimum, trace, countIt, countF = gradient_descent(fs[i], grads[i], start, dichotomy, eps=eps)
            #optimum, trace, countIt, countF = gradient_descent(fs[i], grads[i], start, golden_ratio, eps=eps)

            print(countIt)

            #print(countIt, countF, "|", la.norm(optimum - trace[-2]), "|", la.norm(optimum - realOpts[i]))
            printPlots3D(fs[i], optimum, trace)

"""
for i in range(2, 21):
    f, grad, prF = function_by(i, 100)
    start = np.array([1 for n in range(i)])
    optimum, trace, countIt, countF = nelder_mead(f, start, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(f, grad, start, None, step=1/200, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(f, grad, start, dichotomy, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(f, grad, start, golden_ratio, eps=1e-5)

    print(countIt)


start = np.array([1, 1])
for cond in [500]:#[1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]:
    f, grad, prF = function_by(2, cond)
    #optimum, trace, countIt, countF = nelder_mead(f, start, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(f, grad, start, None, step=1/(2*cond), eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(f, grad, start, dichotomy, eps=1e-5)
    optimum, trace, countIt, countF = gradient_descent(f, grad, start, golden_ratio, eps=1e-5)
    printPlots3D(prF, optimum, trace)
    print(countIt)

func = fs[0], grads[0], fs[0]
start = np.array([1, 0])

for noise in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 1e-4, 1e-6, 1e-8, 0]:
    f, grad, prF = add_noise(*func, noise)

    optimum, trace, countIt, countF = nelder_mead(f, start, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(f, grad, start, None, step=1/3, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(f, grad, start, dichotomy, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(f, grad, start, golden_ratio, eps=1e-5)
    #printPlots3D(prF, optimum, trace)

    print(countIt)
    #printPlots3D(prF, optimum, trace)


fs_mm = [lambda x, y: x**4 + y**4 - x**2 - y**2,
    lambda x, y: x**4 + y**4 - x**2 - y**2 + x,
    lambda x, y: np.cos(x) + np.cos(y)]

grads_mm = [lambda x, y: np.array([4*x**3 - 2*x, 4*y**3 - 2*y]),
    lambda x, y: np.array([4*x**3 - 2*x + 1, 4*y**3 - 2*y]),
    lambda x, y: np.array([-np.sin(x), -np.sin(y)])]

start = np.array([1, 2])
for i in range(2, 3):
    optimum, trace, countIt, countF = nelder_mead(fs_mm[i], start, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(fs_mm[i], grads_mm[i], start, None, step=1.5, eps=1e-5)
    optimum, trace, countIt, countF = gradient_descent(fs_mm[i], grads_mm[i], start, dichotomy, eps=1e-5)
    #optimum, trace, countIt, countF = gradient_descent(fs_mm[i], grads_mm[i], start, golden_ratio, eps=1e-9)
    print(countIt)
    printPlots3D(fs_mm[i], optimum, trace)
"""


#printPlots3D(f, interval=INTERVAL, trace=gradient_descent(f, df, start, golden_ratio)[1])
#printPlots3D(f, interval=INTERVAL, trace=gradient_descent(f, df, start, None, 1e-4)[1])
#printPlots3D(real_prF, trace=opt.minimize(real_f, start, method='Nelder-Mead', options={'return_all': True})["allvecs"])
#print(opt.minimize(f, start, method='Nelder-Mead', options={'return_all': True})["allvecs"])


