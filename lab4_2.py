import math
import optuna
import cmaes
import numpy as np
import numpy.linalg as la

import lab2
import lab3 

fs = [lambda x, y: x**2 + y**2 - x*y,
    lambda x, y: (1 - x)**2 + 100*(y - x**2)**2,
    lambda x, y: np.exp(10*x**2 + y**2),
    lambda x, y: np.log(1 + 100*x**2 + y**2)]

realOpts = [
    np.array([0, 0]),
    np.array([1, 1]),
    np.array([0, 0]),
    np.array([0, 0])
]

samplers = [
    optuna.samplers.RandomSampler(),
    optuna.samplers.TPESampler(),
    optuna.samplers.CmaEsSampler(),
    optuna.samplers.GPSampler(),
    optuna.samplers.NSGAIISampler(),
]

def create_objective(f, dim):
    def objective(trial):
        x = [trial.suggest_float("x" + str(i), -2, 2) for i in range(dim)]
        return f(*x)
    
    return objective


"""trials_range = [5] + list(range(10, 101, 10))

for i in range(len(fs)):
    for sampler_i in range(len(samplers)):
        for trials_i in range(len(trials_range)):
            study = optuna.create_study(sampler=samplers[sampler_i])
            study.optimize(create_objective(fs[i], 2), n_trials=trials_range[trials_i])
            found_opt = np.array(list(study.best_params.values()))
            print(study.best_params, la.norm(found_opt - realOpts[i]))"""


lab2_starts = [
    np.array([1, 1]),
    np.array([1e-2, 1e-2]),
    np.array([math.log(math.pi + 0.5), math.log(2 * math.pi + 0.5)]),
    np.array([0, 0])
]

"""for i in range(0, len(lab2.fs) - 1):
    start = lab2_starts[i]
    def newton(f, start, grad, hessian, step_size, eps=1e-5, N=1000):
        x, trace, res = lab2.newton(f, start, grad, hessian, lab2.const_step_func(step_size), eps=eps, N=N)
        #lab2.printResult(res)
        #lab2.printPlots3D(f, x, trace)
        return res.nit

    def newton_opt(trial):
        step = trial.suggest_float('step', 0, 2)
        return newton(lab2.fs[i], start, lab2.grads[i], lab2.hessians[i], step)

    optuna.logging.disable_default_handler()
    study = optuna.create_study()
    study.optimize(newton_opt, n_trials=200)


    step = study.best_params['step']
    x, trace, res = lab2.newton(lab2.fs[i], start, lab2.grads[i], lab2.hessians[i], lab2.const_step_func(step))
    lab2.printResult(res)
    lab2.printPlots3D(lab2.fs[i], x, trace)

    x, trace, res = lab2.newton(lab2.fs[i], start, lab2.grads[i], lab2.hessians[i], lab2.const_step_func(step - step / 2))
    lab2.printResult(res)
    lab2.printPlots3D(lab2.fs[i], x, trace)

    x, trace, res = lab2.newton(lab2.fs[i], start, lab2.grads[i], lab2.hessians[i], lab2.const_step_func(step + step / 2))
    lab2.printResult(res)
    lab2.printPlots3D(lab2.fs[i], x, trace)

    print(step)"""

def func(x):
    return 10 + 0.8*x

# Параметры
density = 10000
dots_count = 800
dist = 1
radius = 0.1

f = func

X = np.linspace(-dist, dist, density)
Y = np.array([f(x) for x in X])

sample_X, sample_Y = lab3.generate_sample(X, lambda x : f(x), dots_count, radius)
sample = [np.array([sample_X[i], sample_Y[i]]) for i in range(len(sample_X))]

def SGD_batch(batches):
    reg_start = np.array([0, 0])
    reg, memory_u, time_u, epoech, iter, loss = lab3.stochastic_gradient_descent(reg_start, sample, lab3.LinearRegression, lab3.LearningRate.exponential(1, 0.005), lab3.L1(0), batch_size=batches, eps=1e-5, N=1000)
    print(reg)
    return iter

def SGD_step(step):
    reg_start = np.array([0, 0])
    reg, memory_u, time_u, epoech, iter, loss = lab3.stochastic_gradient_descent(reg_start, sample, lab3.LinearRegression, lab3.LearningRate.exponential(step, 0.005), lab3.L1(0), batch_size=round(len(sample) / 2), eps=1e-5, N=1000)
    print(reg)
    return iter

def SGD_opt(trial):
    #batches = trial.suggest_int('batches', 1, len(sample))
    step = trial.suggest_float('step', 0, 2)
    return SGD_step(step)

#optuna.logging.disable_default_handler()
study = optuna.create_study()
study.optimize(SGD_opt, n_trials=200)
best_step = study.best_params['step']
#best_batch = study.best_params['batches']
print(best_step)

