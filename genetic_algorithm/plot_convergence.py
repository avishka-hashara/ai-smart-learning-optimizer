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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    from genetic_algorithm.run_ga import run_ga
    import os
    os.makedirs("docs/figures", exist_ok=True)
    pop, logbook, hof = run_ga()
    plot_convergence(logbook, "docs/figures/ga_convergence.png")
    print("GA convergence plot saved to docs/figures/ga_convergence.png")

