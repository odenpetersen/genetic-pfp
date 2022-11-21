#!/usr/bin/env python3
import numpy as np
from Ndtestfuncs import name_to_func
from matplotlib.image import imsave

def get_best(population, utility, n):
    utilities = np.apply_along_axis(utility,1,population)
    indices = np.argpartition(utilities, -n)[-n:]
    return population[indices], indices

def evolve_inplace(population, utility, n_best, noise=0):
    best, argbest = get_best(population, utility, n_best)
    np.asarray(sorted(population,key=utility))
    for i,_ in enumerate(population):
        if i not in argbest:
            population[i] = best[np.random.randint(best.shape[0]),:] + noise * np.random.normal(size=best.shape[1])

def order_inplace(array, utility):
    #One pass of a bubble sort
    for i in range(len(array)-1):
        if utility(array[i]) > utility(array[i+1]):
            tmp = array[i].copy()
            array[i] = array[i+1].copy()
            array[i+1] = tmp

def run_ga(utility = name_to_func['sphere'], n_population=300, n_params=3, n_generations=300, elite_decay=0.9, noise_decay=0.9):
    elite = noise = 1
    population = np.random.normal(size=(n_population, n_params))

    generations = []
    while len(generations) < n_generations:
        n_best = int(elite * n_population)
        generations.append(population.copy())
        evolve_inplace(population, utility, n_best, noise)
        order_inplace(population, utility) #Aesthetics
        elite = elite*elite_decay
        noise = noise*noise_decay

    return np.apply_along_axis(utility,2,np.asarray(generations))

def make_pfp(filename='pfp.png'):
    #plt.imshow(run_ga(), interpolation='nearest')
    #plt.savefig(filename)
    imsave(filename,run_ga(),cmap='Greys')

if __name__=='__main__':
    make_pfp()

