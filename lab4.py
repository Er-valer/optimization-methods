import math
from turtle import color
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.patheffects as mpath_effects

import numpy as np

COLORMAP_FILL = mcolors.ListedColormap(plt.get_cmap("bone_r")(np.linspace(0, 0.5, 256)))
COLORMAP_CONTOUR = mcolors.ListedColormap(plt.get_cmap("bone_r")(np.linspace(0.25, 0.75, 256)))
COLORMAP_PLOT = mcolors.ListedColormap(plt.get_cmap("gist_heat")(np.linspace(0.25, 0.75, 256)))

PHI = (math.sqrt(5) + 1) / 2

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

    X_trace, Y_trace = zip(*trace)
    color_trace = np.concatenate((np.linspace(0.0, 1.0, round(len(X_trace) / 2)), [1.]*(len(X_trace) - round(len(X_trace) / 2))))

    segments = make_segments(X_trace, Y_trace)
    lc = mcoll.LineCollection(segments, array=color_trace, cmap=COLORMAP_PLOT, path_effects=[mpath_effects.Stroke(capstyle="round")])
    ax = plt.gca()
    ax.add_collection(lc)

    plt.scatter(X_trace, Y_trace, c=color_trace, cmap=COLORMAP_PLOT, zorder=2, marker='.')

    plt.show()

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

###########################################################

class Reproduction:
    def average(fst, snd):
        return (fst + snd) / 2
    
    def crossing_over(fst, snd):
        dim = len(fst)
        crossing = np.random.randint(dim + 1)
        return np.concatenate((fst[:crossing], snd[crossing:]))

class Mutation:
    def __init__(self, dim):
        self.dim = dim

    def constant_interval(self, radius):
        def f(*args):
            return np.array([np.random.uniform(-radius, radius) for _ in range(self.dim)])
        return f

    def exp_tapering_interval(self, radius, k):
        def f(epoch, *args):
            left = -radius * math.exp(-epoch * k)
            right = radius * math.exp(-epoch * k)
            return np.array([np.random.uniform(left, right) for _ in range(self.dim)])
        return f


def norm(fitness):
    worst = min(fitness)
    best = max(fitness)
    if abs(best - worst) < 1e-18:
        return [1] * len(fitness)
    return [(fit - worst) / (best - worst) for fit in fitness]

"""
Поиск максимума функции приспособленности
"""
def genetic_algorithm(f, dim, N=100, mutation_coef=0, survive_coef=1, epochs=1000, eps=1e-5):
    mutation = Mutation(dim).exp_tapering_interval(1e-2, 10. / epochs)
    trace = []
    population = [np.array([1e-2]*dim) for _ in range(N)]
    
    avg_fitness_pr = None
    avg_fitness = None
 
    for epoch in range(epochs):
        # Селекция
        population.sort(key=lambda x : f(*x), reverse=True)
        population = population[:math.floor(N * survive_coef)]
        
        fitness = norm([f(*p) for p in population])
        fitness_sum = sum(fitness)
        probabilities = [fitness[i] / fitness_sum for i in range(len(population))]

        # Проверка критерия остановки
        trace.append(population[0].copy())
        avg_fitness_pr = avg_fitness
        avg_fitness = sum([f(*p) for p in population])/len(population)
        
        #print(epoch, f(*population[0]), avg_fitness, avg_fitness_pr)
        if avg_fitness_pr != None and abs(avg_fitness - avg_fitness_pr) < eps:
            break
        
        # Размножение
        new_population = []
        for _ in range(N - len(population)):
            fst = population[np.random.choice(range(len(population)), p=probabilities)]
            snd = population[np.random.choice(range(len(population)), p=probabilities)]
            new_population.append(Reproduction.crossing_over(fst, snd))
        population = population + new_population

        # Мутация
        for _ in range(math.floor(N * mutation_coef)):
            population[np.random.randint(N)] += mutation(epoch)
    
    return population[0], trace

def f(x, y):
    return np.log(1 + 100*x**2 + y**2)

def f_neg(f):
    return lambda *x: -f(*x) 

optimum, trace = genetic_algorithm(f_neg(f), 2, N=100, mutation_coef=0.5, survive_coef=0.9, epochs=1000)
print(len(trace) * 100)
print(f(*optimum))
printPlots3D(f, optimum, trace)