import math

def expected_score_gain(hours, k=0.15):
    """
    Diminishing-returns model: score improvement per topic
    flattens out as hours increase (concave, not linear).
    """
    return 1 - math.exp(-k * hours)

def fitness_function(hours_allocated, *args, **kwargs):
    """
    Supports both Member 3's initial test signature:
        fitness_function(hours_allocated, gain_fn, urgency_weights)
    and the production signature:
        fitness_function(hours_allocated, priority, predicted, min_h, max_h, ...)
    """
    if len(args) >= 2 and callable(args[0]):
        gain_fn = args[0]
        urgency_weights = args[1]
        return sum(
            weight * gain_fn(h)
            for h, weight in zip(hours_allocated, urgency_weights)
        )
    
    from src.optimizer.fitness import fitness_function as prod_fit
    return prod_fit(hours_allocated, *args, **kwargs)
