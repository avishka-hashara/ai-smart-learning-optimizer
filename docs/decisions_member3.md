# Member 3 Decisions Log: Genetic Algorithm Optimizer

| Decision / Parameter | Chosen Setting | Plain-English Reasoning (Handbook Reference) |
| :--- | :--- | :--- |
| **Chromosome Representation** | Real-valued vector, length $n$ | Revision hours are continuous quantities. Binary strings impose an artificial quantization and break arithmetic crossover semantics. |
| **Population Size** | 80 individuals | Provides sufficient genetic diversity for 5–8 subjects while completing 100 generations in under 0.5s for interactive UI use. |
| **Generations** | 100 | Ensures convergence plateau is reliably reached without premature termination. |
| **Selection Operator** | Tournament selection (size 3) | Moderate selection pressure that is scale-invariant (unlike roulette wheel), ensuring robustness across varying study budgets and priorities. |
| **Crossover Operator** | Blend crossover ($\alpha = 0.5$, rate $0.8$) | Real-valued arithmetic blending generates offspring in the interior of parent solutions, staying close to the feasible sum hyperplane. |
| **Mutation Operator** | Gaussian mutation ($\sigma = 1.0$, rate $0.2$) | Introduces controlled exploratory perturbations to escape local optima without destroying good allocations. |
| **Elitism** | Preserve top 2 individuals | Guarantees that the best solution found is never lost to random operator destruction; keeps the convergence curve non-decreasing. |
| **Diminishing Returns ($\tau$)** | $\tau = 7.0$ hours | Parameter in $1 - e^{-h/\tau}$ where ~63.2% of attainable score gain is achieved by 7 hours of study per module. |
| **Headroom Model** | $\max(0, 100 - \text{predicted})$ | Caps attainable score gain to available marks, preventing high-scoring subjects (e.g. 90%) from claiming unnecessary hours. |
| **Constraint Handling** | Bounded 5-iteration repair | Alternates budget scaling and bound clipping up to 5 times, followed by residual distribution, ensuring $\sum h_i = \text{budget} \pm 0.01$. |
| **Penalty Terms** | Quadratic ($\lambda = 20.0$) | Heavily penalizes solutions violating lower or upper limits ($\min\_h, \max\_h$) to steer search toward feasible space. |
| **Random Seed** | 42 | Ensures deterministic reproducibility of reported benchmark comparisons. |
