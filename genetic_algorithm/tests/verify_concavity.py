from genetic_algorithm.fitness import fitness_function, expected_score_gain
 
 
def verify_concavity(total_hours=10, urgency_weights=(0.7, 0.3)):
    print(f"{'H1':>4} {'H2':>4} {'Fitness':>10}")
    for h1 in range(total_hours + 1):
        h2 = total_hours - h1
        fit = fitness_function([h1, h2], expected_score_gain, urgency_weights)
        print(f"{h1:>4} {h2:>4} {fit:>10.4f}")
 
 
if __name__ == "__main__":
    verify_concavity()

