import matplotlib.pyplot as plt
 
 
def plot_convergence(logbook, save_path="convergence.png"):
    gens = logbook.select("gen")
    best_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")
 
    plt.plot(gens, best_fitness, label="Best fitness")
    plt.plot(gens, avg_fitness, label="Average fitness", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Convergence: Generation vs Best Fitness")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

