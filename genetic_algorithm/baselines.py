def equal_split(num_topics, total_hours):
    """Split hours evenly across all topics."""
    return [total_hours / num_topics] * num_topics

