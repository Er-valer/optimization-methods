import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import numpy.linalg as la

import scipy.optimize as opt

COLORMAP_FILL = ListedColormap(plt.get_cmap("bone_r")(np.linspace(0, 0.5, 256)))
COLORMAP_CONTOUR = ListedColormap(plt.get_cmap("bone_r")(np.linspace(0.25, 0.75, 256)))
PHI = (math.sqrt(5) + 1) / 2

class OptimizeResult():
    def __init__(self):
        self.x = None
        self.success = False
        self.trace = []
        self.nit = 0
        self.nfev = 0
        self.njev = 0
        self.nhev = 0
    
    def __add__(self, o):
        res = OptimizeResult()
        res.nit = self.nit + o.nit
        res.nfev = self.nfev + o.nfev
        res.njev = self.njev + o.njev
        res.nhev = self.nhev + o.nhev
        return res
    
    def __repr__(self):
        return '\n'.join([
            f"success : {self.success}",
            f"iterations : {self.nit}", 
            f"function calls : {self.nfev}", 
            f"gradient calls : {self.njev}", 
            f"hessian calls : {self.nhev}"
        ])

def printResult(opt_res):
    print(opt_res.nit, "|", opt_res.nfev, "|", opt_res.njev, "|", opt_res.nhev)

def printPlots2D(f, interval, trace, delta = 1e-5):
    x = np.arange(interval[0], interval[1], delta)
    X = x

    Y = f(X)

    plt.plot(X, Y)
    plt.plot(*zip(*trace))

    plt.show()

def printPlotDefault(f, dots=1e3):
    intervalX = (-5, 5)
    intervalY = (-5, 5)

    x = np.arange(intervalX[0], intervalX[1], (intervalX[1]-intervalX[0])/dots)
    y = np.arange(intervalY[0], intervalY[1], (intervalY[1]-intervalY[0])/dots)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)

    plt.contour(X, Y, Z, levels=20, cmap=COLORMAP_CONTOUR)
    plt.contourf(X, Y, Z, levels=20, cmap=COLORMAP_FILL)
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm",
                       linewidth=0, antialiased=False)
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

    contour = plt.contour(X, Y, Z, levels=20, cmap=COLORMAP_CONTOUR)
    plt.contourf(X, Y, Z, levels=20, cmap=COLORMAP_FILL)
    plt.clabel(contour)
    plt.plot(*zip(*trace), color="red", marker='.')

    plt.show()

def golden_ratio(f, interval, N = 100, eps=1e-3):
    res = OptimizeResult()

    l, r = interval
    x1, x2 = r - (r - l)/PHI, l + (r - l)/PHI
    
    fx1, fx2 = f(x1), f(x2)
    res.nfev += 2

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

        res.nit += 1
        res.nfev += 1

        if ((r - l)/2 < eps):
            break
    
    return (l + r)/2, res

def const_step_func(step):
    def func(*_):
        res = OptimizeResult()
        res.nit += 1
        return step, res
    
    return func

def find_interval(f, step = 1e-1):
    s = step
    while f(s) < f(0):
        s += step
    return (0, s)

def best_step_by_onedim_method(method):
    def func(f, x, _, direction):
        g = lambda s: f(*(x + s * direction))
        return method(g, find_interval(g))
    
    return func

def best_step_by_wolfe(start, eps1 = 1e-4, eps2 = 0.9, theta1 = 0.9, theta2 = 1.1):
    def func(f, x, grad, direction):
        res = OptimizeResult()
        def armijo(a):
            res.nfev += 2
            res.njev += 1
            return f(*(x + a*direction)) - f(*x) <= eps1*a*(grad(*x) @ direction)
        
        def curvature(a):
            res.njev += 2
            return grad(*(x + a*direction)) @ direction >= eps2*(grad(*x) @ direction)

        a = start
        while True:
            res.nit += 1
            if not armijo(a):
                a *= theta1
            elif not curvature(a):
                a *= theta2

            return a, res
    
    return func
        
def gradient_descent(f, grad, start, method, N=100000, eps=1e-6):
    res = OptimizeResult()
    steps_res = OptimizeResult()

    x = start
    x_pr = None

    trace = [x]

    for _ in range(N):
        direction = -grad(*x)

        res.nit += 1
        res.njev += 1

        best_step, res_step = method(f, x, grad, direction)
        lr = best_step
        steps_res += res_step
        
        x_pr = x
        x = x + lr * direction

        trace.append(x)

        if (la.norm(x - x_pr) <= eps):
            res.success = True
            break

    return x, trace, res + steps_res

def nelder_mead(f, start, N = 100000, eps = 1e-9):
    f_ = lambda x: f(*x)
    result = opt.minimize(f_, start, method='Nelder-Mead', options={'return_all': True, 'maxiter': N, 'xatol': eps})

    res = OptimizeResult()
    res.success = result.success
    res.nit = result.nit
    res.nfev = result.nfev

    return result.x, result.allvecs, res

####################-- Hewton Method --#######################

def newton(f, start, grad, hessian, step, eps = 1e-9, N = 100000):
    res = OptimizeResult()
    steps_res = OptimizeResult()

    x_pr = None
    x = start

    trace = [x]

    for _ in range(N):
        res.nit += 1
        res.njev += 1
        res.nhev += 1

        direction = -(la.inv(hessian(*x)) @ grad(*x))

        x_pr = x
        best_step, step_res = step(f, x, grad, direction)
        steps_res += step_res

        x = x + best_step * direction

        trace.append(x)

        if (la.norm(x - x_pr) <= eps):
            res.success = True
            break
        
    res.trace = trace
    res.x = x

    return x, trace, res + steps_res

def newton_cg(f, start, grad = None, hessian = None, eps = 1e-9, N = 100000):
    f_ = lambda x: f(*x)
    grad_ = lambda x: grad(*x)

    if hessian:
        hessian_ = lambda x: hessian(*x)

    result = opt.minimize(f_, start, method='Newton-CG', jac = grad_, options={'return_all': True, 'maxiter': N, 'xtol': eps})
    return result.x, result.allvecs, result

def bfgs(f, start, grad = None, eps = 1e-9, N = 100000):
    f_ = lambda x: f(*x)
    grad_ = None

    if grad:
        grad_ = lambda x: grad(*x)
    
    result = opt.minimize(f_, start, method='BFGS', jac = grad_, options={'return_all': True, 'maxiter': N, 'gtol': eps})

    res = OptimizeResult()
    res.success = result.success
    res.nit = result.nit
    res.nfev = result.nfev
    res.njev = result.njev

    return result.x, result.allvecs, res

##################################################

methods = [
    const_step_func(1),
    best_step_by_onedim_method(golden_ratio),
    best_step_by_wolfe(1)
]

fs = [
    lambda x, y: np.exp(10*x**2 + y**2),
    lambda x, y: np.log(1 + 100*x**2 + y**2),
    lambda x, y: np.cos(np.exp(x)) * np.cos(np.exp(y)),
    lambda x, y: (1 - x)**2 + 100*(y - x**2)**2
]

grads = [
    lambda x, y: np.array([
        20*x*np.exp(10*x**2 + y**2),
        2*y*np.exp(10*x**2 + y**2)
    ]),
    lambda x, y: np.array([
        200*x / (1 + 100*x**2 + 100*y**2),
        2*y / (1 + 100*x**2 + 100*y**2)
    ]),
    lambda x, y: np.array([
        -np.exp(x) * np.sin(np.exp(x)) * np.cos(np.exp(y)), 
        -np.exp(y) * np.sin(np.exp(y)) * np.cos(np.exp(x))
    ]),
    lambda x, y: np.array([2*(-1 + x + 200*x**3 - 200*x*y),  200*(-x**2 + y)])
]

hessians = [
    lambda x, y: np.array([
        [
            (400*x**2 + 20)*np.exp(10*x**2 + y**2),
            40*x*y*np.exp(10*x**2 + y**2)
        ],
        [
            40*x*y*np.exp(10*x**2 + y**2),
            (4*y**2 + 2)*np.exp(10*x**2 + y**2)
        ]]),
    lambda x, y: np.array([
        [
            200 / (1 + 100*x**2 + 100*y**2) - 40000*x**2 / (1 + 100*x**2 + 100*y**2)**2,
            -400*x*y / (1 + 100*x**2 + 100*y**2)**2
        ], 
        [
            -400*x*y / (1 + 100*x**2 + 100*y**2)**2,
            2 / (1 + 100*x**2 + 100*y**2) - 4*y**2 / (1 + 100*x**2 + 100*y**2)**2
        ]
    ]),
    lambda x, y: np.array([
        [
            -np.exp(2*x) * np.cos(np.exp(x)) * np.cos(np.exp(y)) - np.exp(x) * np.sin(np.exp(x)) * np.cos(np.exp(y)),
            np.exp(x + y) * np.sin(np.exp(x)) * np.sin(np.exp(y))
        ],
        [
            np.exp(x + y) * np.sin(np.exp(x)) * np.sin(np.exp(y)),
            -np.exp(2*y) * np.cos(np.exp(x)) * np.cos(np.exp(y)) - np.exp(y) * np.sin(np.exp(y)) * np.cos(np.exp(x))
        ]
    ]),
    lambda x, y: np.array([
        [-400*(y - x**2) + 800*x**2 + 2, -400*x],
        [-400*x, 200]
    ])
]

starts = [
    [
        np.array([2e-2, 3e-1]),
        np.array([1e-1, 1e-1]),
        np.array([1, 1]),
    ],
    [
        np.array([1e-3, 1e-1]),
        np.array([1e-2, 1e-2]),
        np.array([5e-2, 0]),
    ],
    [
        np.array([math.log(math.pi + 1), math.log(2 * math.pi + 1)]),
        np.array([math.log(math.pi + 0.5), math.log(2 * math.pi + 0.5)]),
        np.array([math.log(math.pi - 0.5), math.log(2 * math.pi + 0.5)]),
        np.array([math.log(math.pi + 1), math.log(2 * math.pi)])
    ],
    [
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([0, 1]),
    ]
]

for f in fs:
    printPlotDefault(f)

for method in methods:
    for i in range(len(fs)):
        for start in starts[i]:
            x, trace, res = newton(fs[i], start, grads[i], hessians[i], method)
            printResult(res)
            printPlots3D(fs[i], x, trace)

            x, trace, res = newton_cg(fs[i], start, grads[i], hessians[i])
            printResult(res)
            printPlots3D(fs[i], x, trace)

            x, trace, res = gradient_descent(fs[i], grads[i], start, methods[1])
            printResult(res)
            printPlots3D(fs[i], x, trace)
            
            x, trace, res = bfgs(fs[i], start)
            printResult(res)
            printPlots3D(fs[i], x, trace)

            x, trace, res = bfgs(fs[i], start, grad=grads[i])
            printResult(res)
            printPlots3D(fs[i], x, trace)

            x, trace, res = nelder_mead(fs[i], start)
            printResult(res)
            printPlots3D(fs[i], x, trace)
