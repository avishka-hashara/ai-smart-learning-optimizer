import math
 
 
def expected_score_gain(hours, k=0.15):
    """
    Diminishing-returns model: score improvement per topic
    flattens out as hours increase (concave, not linear).
    """
    return 1 - math.exp(-k * hours)
 
 
def fitness_function(hours_allocated, gain_fn, urgency_weights):
    """
    Pure function: total weighted expected score improvement.
 
    hours_allocated : list[float]  hours per topic
    gain_fn         : callable     e.g. expected_score_gain
    urgency_weights : list[float]  fuzzy urgency/priority per topic (0-1)
    """
    return sum(
        weight * gain_fn(hours)
        for hours, weight in zip(hours_allocated, urgency_weights)
    )
