import numpy as np

N_BEST_DENOMINATOR = 2
NORMAL_DENOMINATOR = 20


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
