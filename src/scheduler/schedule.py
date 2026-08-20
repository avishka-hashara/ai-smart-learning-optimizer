def build_schedule(hours, subjects, days, daily_limits, peak_window, exam_dates=None):
    """Stage 4 Stub: Returns a fake schedule block for each subject."""
    schedule = []
    for subj in subjects:
        schedule.append({
            'day': 'Monday',
            'date': '2026-10-12',
            'start': '09:00',
            'end': '10:00',
            'subject': subj,
            'minutes': 60,
            'type': 'study',
            'reason': 'Stub placement'
        })
    return schedule