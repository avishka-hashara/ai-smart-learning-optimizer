import random
from deap import base, creator, tools
 
NUM_TOPICS = 5
TOTAL_HOURS = 20
 
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
 
toolbox = base.Toolbox()
toolbox.register("attr_hours", random.uniform, 0, TOTAL_HOURS)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual,
    toolbox.attr_hours, n=NUM_TOPICS
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

from genetic_algorithm.fitness import fitness_function, expected_score_gain
from genetic_algorithm.repair import repair
 
 
def evaluate(individual, urgency_weights, total_hours=TOTAL_HOURS):
    repaired = repair(individual, total_hours)
    score = fitness_function(repaired, expected_score_gain, urgency_weights)
    return (score,)  
 
toolbox.register("evaluate", evaluate, urgency_weights=[0.7, 0.5, 0.3, 0.6, 0.4])

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

def mutate_and_repair(individual, mu=0, sigma=2, indpb=0.2, total_hours=TOTAL_HOURS):
    tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb)
    individual[:] = repair(individual, total_hours)
    return (individual,)  # DEAP expects a tuple
 
 
toolbox.register("mutate", mutate_and_repair)
