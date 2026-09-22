from datetime import datetime
import numpy as np


def build_schedule(hours, subjects, days, daily_limits, peak_window, exam_dates=None):
    """
    Rule-Based Study Scheduler.

    Creates study sessions using:
    - allocated hours
    - available days
    - daily study limit
    - preferred study time
    - exam dates when available
    """

    schedule = []

    # No subjects means no schedule
    if not subjects:
        return schedule

    # Convert number of days into day names
    if isinstance(days, int):
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ][:days]
    else:
        day_names = list(days)

    if not day_names:
        return schedule

    # ---------------------------------------------------------
    # Convert allocated hours safely to a 1-D numeric array
    # ---------------------------------------------------------
    try:
        hours = np.asarray(hours, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(
            "Stage 4 Error: allocated hours must contain numeric values."
        )

    # Make sure there is one value for each subject
    if len(hours) != len(subjects):
        raise ValueError(
            f"Stage 4 Error: received {len(hours)} hour values "
            f"for {len(subjects)} subjects."
        )

    # Convert allocated hours into minutes
    subject_minutes = [
        int(round(float(h) * 60))
        for h in hours
    ]

    # Daily study limit
    daily_limit_minutes = int(float(daily_limits) * 60)

    if daily_limit_minutes <= 0:
        return schedule

    # Get preferred starting time
    start_time = "09:00"

    if isinstance(peak_window, str) and "-" in peak_window:
        start_time = peak_window.split("-")[0].strip()

    # Subject order
    subject_order = list(range(len(subjects)))

    # Give priority to earlier exams
    if exam_dates:
        try:
            subject_order.sort(
                key=lambda i: datetime.strptime(
                    str(exam_dates.get(subjects[i], "2099-12-31")),
                    "%Y-%m-%d"
                )
            )
        except (ValueError, AttributeError):
            pass

    # Track current day
    current_day = 0
    minutes_used_today = 0

    # Create study sessions
    for index in subject_order:

        remaining_minutes = subject_minutes[index]

        while remaining_minutes > 0:

            # Move to next day when daily limit is reached
            if minutes_used_today >= daily_limit_minutes:
                current_day += 1
                minutes_used_today = 0

            # Stop when all available days are used
            if current_day >= len(day_names):
                return schedule

            # Maximum session = 60 minutes
            session_minutes = min(
                60,
                remaining_minutes,
                daily_limit_minutes - minutes_used_today
            )

            # Convert starting time to minutes
            start_hour, start_minute = map(
                int,
                start_time.split(":")
            )

            start_total = (
                start_hour * 60
                + start_minute
                + minutes_used_today
            )

            end_total = start_total + session_minutes

            start_h = (start_total // 60) % 24
            start_m = start_total % 60

            end_h = (end_total // 60) % 24
            end_m = end_total % 60

            schedule.append({
                "day": day_names[current_day],
                "date": day_names[current_day],
                "start": f"{start_h:02d}:{start_m:02d}",
                "end": f"{end_h:02d}:{end_m:02d}",
                "subject": subjects[index],
                "minutes": session_minutes,
                "type": "study",
                "reason": "Rule-based scheduling"
            })

            remaining_minutes -= session_minutes
            minutes_used_today += session_minutes

    return schedule