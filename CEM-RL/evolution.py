import numpy as np
#
# N_BEST_DENOMINATOR = 2
# NORMAL_DENOMINATOR = 20
#
#
# def evolve(orgs, fitness):
#     fitness = np.array(fitness)
#     fitness *= -1
#     idx_sorted = np.argsort(fitness)
#     idx_best = idx_sorted[: len(idx_sorted) // N_BEST_DENOMINATOR]
#
#     new_orgs = []
#
#     for _ in range(len(orgs)):
#         idx = np.random.choice(idx_best, 2)
#         phi = np.random.rand()
#         new_org = orgs[idx[0]] * phi + orgs[idx[1]] * (1 - phi)
#         new_org = new_org + np.random.normal(0, np.abs(new_org) / NORMAL_DENOMINATOR)
#         new_orgs.append(new_org)
#
#     return new_orgs


# # Parametry
# TOURNAMENT_SIZE = 3
# NORMAL_DENOMINATOR = 10
#
#
# def tournament_selection(orgs, fitness, tournament_size):
#     selected_indices = []
#     for _ in range(tournament_size):
#         idx = np.random.randint(0, len(orgs))
#         selected_indices.append(idx)
#     best_idx = selected_indices[np.argmax([fitness[i] for i in selected_indices])]
#     return best_idx
#
#
# def evolve(orgs, fitness):
#     new_orgs = []
#
#     for _ in range(len(orgs)):
#         idx1 = tournament_selection(orgs, fitness, TOURNAMENT_SIZE)
#         idx2 = tournament_selection(orgs, fitness, TOURNAMENT_SIZE)
#
#         phi = np.random.rand()
#         new_org = orgs[idx1] * phi + orgs[idx2] * (1 - phi)
#         new_org = new_org + np.random.normal(0, np.abs(new_org) / NORMAL_DENOMINATOR)
#         new_orgs.append(new_org)
#
#     return new_orgs

import numpy as np

TOURNAMENT_SIZE = 3  #
NORMAL_DENOMINATOR = 10  #


def roulette_selection(orgs, fitness):
    total_fitness = sum(fitness)
    if total_fitness == 0:
        return np.random.randint(0, len(orgs))
    pick = np.random.rand() * total_fitness
    current = 0
    for i, f in enumerate(fitness):
        current += f
        if current > pick:
            return i
    return len(fitness) - 1


def single_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def gaussian_mutation(org, sigma=1.0):
    return org + np.random.normal(0, sigma, size=org.shape)


def evolve(orgs, fitness):
    new_orgs = []

    while len(new_orgs) < len(orgs):
        idx1 = roulette_selection(orgs, fitness)
        idx2 = roulette_selection(orgs, fitness)

        parent1, parent2 = orgs[idx1], orgs[idx2]
        child1, child2 = single_point_crossover(parent1, parent2)

        child1 = gaussian_mutation(child1, sigma=np.abs(child1) / NORMAL_DENOMINATOR)
        child2 = gaussian_mutation(child2, sigma=np.abs(child2) / NORMAL_DENOMINATOR)

        new_orgs.append(child1)
        if len(new_orgs) < len(orgs):
            new_orgs.append(child2)

    return new_orgs

