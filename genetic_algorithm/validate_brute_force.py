from genetic_algorithm.fitness import fitness_function, expected_score_gain
 
 
def brute_force_best(total_hours, urgency_weights, step=1):
    best_fit, best_alloc = -1, None
    for h1 in range(0, total_hours + 1, step):
        h2 = total_hours - h1
        fit = fitness_function([h1, h2], expected_score_gain, urgency_weights)
        if fit > best_fit:
            best_fit, best_alloc = fit, (h1, h2)
    return best_alloc, best_fit
 
 
if __name__ == "__main__":
    alloc, fit = brute_force_best(total_hours=20, urgency_weights=[0.7, 0.4])
    print("Brute-force best allocation:", alloc, "fitness:", fit)
   

