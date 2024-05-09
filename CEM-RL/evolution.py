import numpy as np


def tell(self, solutions, scores):
    """
    Updates the distribution
    """
    scores = np.array(scores)
    scores *= -1
    idx_sorted = np.argsort(scores)

    old_mu = self.mu
    self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
    self.mu = self.weights @ solutions[idx_sorted[: self.parents]]

    z = solutions[idx_sorted[: self.parents]] - old_mu
    self.cov = 1 / self.parents * self.weights @ (z * z) + self.damp * np.ones(
        self.num_params
    )

    self.elite = solutions[idx_sorted[0]]
    self.elite_score = scores[idx_sorted[0]]
    print(self.cov)


N_BEST_DENOMINATOR = 2
NORMAL_DENOMINATOR = N_BEST_DENOMINATOR * 10


def evolve(orgs, fitness):
    fitness = np.array(fitness)
    fitness *= -1
    idx_sorted = np.argsort(fitness)
    idx_best = idx_sorted[: len(idx_sorted) // N_BEST_DENOMINATOR]

    new_orgs = []

    for _ in range(len(orgs)):
        idx = np.random.choice(idx_best, 2)
        phi = np.random.rand()
        new_org = orgs[idx[0]] * phi + orgs[idx[1]] * (1 - phi)
        new_org = new_org + np.random.normal(0, np.abs(new_org) / NORMAL_DENOMINATOR)
        new_orgs.append(new_org)

    return new_orgs
