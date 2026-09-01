import random
from deap import algorithms, tools
from genetic_algorithm.deap_setup import toolbox
 
 
def run_ga(pop_size=50, generations=40, cxpb=0.6, mutpb=0.3, seed=42):
    random.seed(seed)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
 
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", max)
    stats.register("avg", lambda vals: sum(vals) / len(vals))
 
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=cxpb, mutpb=mutpb,
        ngen=generations, stats=stats, halloffame=hof, verbose=True
    )
    return pop, logbook, hof
 
 
if __name__ == "__main__":
    pop, logbook, hof = run_ga()
    print("Best allocation:", hof[0])
    print("Best fitness:", hof[0].fitness.values[0])
