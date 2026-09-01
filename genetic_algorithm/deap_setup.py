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

