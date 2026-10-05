def repair(hours, total_hours, min_h=0, max_h=None):
    """
    Clip each topic's hours into [min_h, max_h], then rescale
    so the allocation sums back to total_hours.
    """
    if max_h is None:
        max_h = total_hours

    clipped = [max(min_h, min(max_h, h)) for h in hours]
    current_total = sum(clipped)

    if current_total == 0:
        return [total_hours / len(hours)] * len(hours)

    scale = total_hours / current_total
    return [h * scale for h in clipped]
