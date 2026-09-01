def equal_split(num_topics, total_hours):
    """Split hours evenly across all topics."""
    return [total_hours / num_topics] * num_topics

def proportional_split(urgency_weights, total_hours):
    """Split hours in proportion to urgency weight."""
    total_weight = sum(urgency_weights)
    return [total_hours * (w / total_weight) for w in urgency_weights]

def greedy_split(urgency_weights, total_hours, step=1, decay=0.9):
    """Greedily give hours to the highest-urgency topic, decaying
    that topic's weight after each allocation so others get a turn."""
    weights = list(urgency_weights)
    hours = [0] * len(weights)
    remaining = total_hours
 
    while remaining > 0:
        best_topic = max(range(len(weights)), key=lambda i: weights[i])
        allocation = min(step, remaining)
        hours[best_topic] += allocation
        remaining -= allocation
        weights[best_topic] *= decay
 
    return hours


