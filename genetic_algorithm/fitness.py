"""
Bridge module pointing to production src.optimizer.fitness.
Preserved for backward compatibility with Member 3 branch commits.
"""
from src.optimizer.fitness import (
    expected_score_gain,
    fitness_function,
    DEFAULT_TAU,
    DEFAULT_PENALTY_LAMBDA
)
