import random
import numpy as np
from deap import base, creator, tools
from src.optimizer.fitness import fitness_function
from src.optimizer.repair import repair

# Ensure DEAP classes are created once
if not hasattr(creator, "OptimizerFitnessMax"):
    creator.create("OptimizerFitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "OptimizerIndividual"):
    creator.create("OptimizerIndividual", list, fitness=creator.OptimizerFitnessMax)

def run_ga(priority, predicted, budget, min_h, max_h,
           pop_size=80, generations=100, cxpb=0.8, mutpb=0.2,
           elitism_size=2, seed=42, tau=7.0):
    """
    Dynamic Genetic Algorithm for Subject Study Hour Optimization (Handbook Part 3).

    Parameters
    ----------
    priority : array-like of shape (n,)
    predicted : array-like of shape (n,)
    budget : float
    min_h : array-like of shape (n,)
    max_h : array-like of shape (n,)
    pop_size : int, default 80
    generations : int, default 100
    cxpb : float, default 0.8
    mutpb : float, default 0.2
    elitism_size : int, default 2
    seed : int, default 42
    tau : float, default 7.0

    Returns
    -------
    best_chromosome : np.ndarray
    best_fitness : float
    history : list of float (best fitness per generation)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = len(priority)
    prio = np.asarray(priority, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    min_b = np.asarray(min_h, dtype=float)
    max_b = np.asarray(max_h, dtype=float)

    toolbox = base.Toolbox()

    def create_individual():
        # Initialize with random values between min_b and max_b
        raw = [random.uniform(mn, mx) for mn, mx in zip(min_b, max_b)]
        repaired = repair(raw, budget, min_b, max_b)
        return creator.OptimizerIndividual(repaired.tolist())

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(ind):
        # Always evaluate a strictly repaired version
        repaired = repair(ind, budget, min_b, max_b)
        fit = fitness_function(repaired, prio, pred, min_b, max_b, tau=tau)
        return (fit,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    # Initialize population
    population = toolbox.population(n=pop_size)

    # Seed initial population with heuristic baselines to jumpstart search
    from src.optimizer.baselines import equal_split, proportional_split, greedy_split
    population[0] = creator.OptimizerIndividual(equal_split(n, budget, min_b, max_b).tolist())
    population[1] = creator.OptimizerIndividual(proportional_split(prio, budget, min_b, max_b).tolist())
    population[2] = creator.OptimizerIndividual(greedy_split(prio, pred, budget, min_b, max_b, tau=tau).tolist())

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(elitism_size)
    hof.update(population)

    history = [float(hof[0].fitness.values[0])]

    for gen in range(1, generations + 1):
        # Select parents for next generation (reserving slots for elitism)
        offspring_count = pop_size - elitism_size
        offspring = toolbox.select(population, offspring_count)
        offspring = list(map(toolbox.clone, offspring))

        # Apply Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply Mutation and immediate repair
        for mutant in offspring:
            if random.random() < mutpb:
                tools.mutGaussian(mutant, mu=0.0, sigma=1.0, indpb=0.25)
                del mutant.fitness.values

            # Repair mutant/crossover individuals
            repaired = repair(mutant, budget, min_b, max_b)
            mutant[:] = repaired.tolist()

        # Evaluate un-evaluated offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # Elitism: preserve top individuals
        elite = [toolbox.clone(ind) for ind in hof]

        # Form next generation
        population[:] = elite + offspring
        hof.update(population)

        history.append(float(hof[0].fitness.values[0]))

    best_individual = np.array(repair(hof[0], budget, min_b, max_b), dtype=float)
    best_fitness = float(hof[0].fitness.values[0])

    return best_individual, best_fitness, history
