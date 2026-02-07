import os
import sys
import numpy as np
from scipy.stats import entropy

# Get the absolute path to the directory where fairamga.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairamga.py
sys.path.append(os.path.join(base_path, "../"))
from tutorial.adaptive_mutation import AdaptiveMutation


class GA:
    def __str__(self):
        return "GA"

    def __init__(self, pop_size, dna_size, bound, discrimination_check, diversity_method='hamming_stratified',
                 crossover_rate=0.5, mutation_rate=0.05, sigma=0.05, lambda_=10, sampling_threshold=50,
                 evolution_strategy='default', adaptation_scheme='linear'):
        self.gen = 0
        self.N = pop_size
        self.n = dna_size
        self.bound = np.array(bound)
        self.fitness_func = discrimination_check
        self.diversity_method = diversity_method
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.sigma = sigma
        self.lambda_ = lambda_

        self.population = self.generate_initial_population()
        self.best_fitness_history = []
        self.mutation_rate_history = [[self.gen, self.mr]]
        self.elite_size = int(self.N * 0.1)  # 10% elitism
        self.tournament_size = max(5, int(self.N * 0.05))
        self.evolution_strategy = evolution_strategy


        self.mutation_adapter = AdaptiveMutation(
            mr_min=0.05,
            mr_max=0.5,
            sampling_threshold=sampling_threshold,
            bounds=bound,
            adaptation_scheme=adaptation_scheme
        )
        self.diversity = 0

    def generate_initial_population(self):
        return np.random.uniform(
            low=self.bound[:, 0],
            high=self.bound[:, 1],
            size=(self.N, self.n)
        )

    def evaluate_fitness(self):
        return self.fitness_func(self.population)

    def select(self, fitness):

        # Elitism selection Select the best individuals from the population (based on fitness) and keep them
        # unchanged in the next generation. The number of selected individuals depends on the elitism size (10% of
        # the population). The selected individuals are then combined with the rest of the population to form the
        # next generation.
        elite_indices = np.argsort(fitness)[-self.elite_size:]
        elite = self.population[elite_indices]

        # Tournament selection for the rest
        selected = []
        for _ in range(self.N - self.elite_size):
            tournament = np.random.choice(np.arange(self.N), self.tournament_size, replace=False)
            winner = tournament[np.argmax(fitness[tournament])]
            selected.append(self.population[winner])

        self.population = np.vstack((elite, selected))
    
    def uniform_crossover(self):
        """Uniform crossover applied in-place to the whole population.
        We use a uniform crossover operator with probability 0.8 applied at the gene level.
        Additionally, when using feature importance guidance, crossover is restricted to features
        with importance exceeding a predefined threshold."""
        new_population = []
        indices = np.arange(self.N)
        np.random.shuffle(indices)   # random pairing

        for i in range(0, self.N, 2):
            p1 = self.population[indices[i]].copy()
            p2 = self.population[indices[i+1]].copy()

            # children initially copies of parents
            c1, c2 = p1.copy(), p2.copy()

            # uniform crossover: loop gene by gene
            for j in range(self.n):
                if np.random.rand() < self.cr:
                    c1[j], c2[j] = c2[j], c1[j]

            new_population.append(c1)
            new_population.append(c2)

        self.population = np.array(new_population)

    def hux_crossover(self):
        """Half Uniform Crossover - only exchanges at differing positions."""
        new_population = []
        indices = np.arange(self.N)
        np.random.shuffle(indices)

        for i in range(0, self.N, 2):
            p1 = self.population[indices[i]].copy()
            p2 = self.population[indices[i + 1]].copy()

            # Find positions where parents differ
            diff_positions = np.where(p1 != p2)[0]
            n_diff = len(diff_positions)

            if n_diff > 0:
                # Select half of the differing positions to swap
                n_swap = n_diff // 2
                swap_positions = np.random.choice(diff_positions, n_swap, replace=False)

                # Create children
                c1, c2 = p1.copy(), p2.copy()

                # Swap at selected positions
                c1[swap_positions] = p2[swap_positions]
                c2[swap_positions] = p1[swap_positions]
            else:
                # Parents are identical, children are copies
                c1, c2 = p1.copy(), p2.copy()

            new_population.append(c1)
            new_population.append(c2)

        self.population = np.array(new_population)

    def gaussian_mutate(self):
        for i in range(self.elite_size, self.N):  # Don't mutate elite individuals
            for j in range(self.n):
                if np.random.rand() < self.mr:
                    delta = (self.bound[j, 1] - self.bound[j, 0])*0.1
                    self.population[i, j] += np.random.uniform(-delta, delta)
                    self.population[i, j] = np.clip(self.population[i, j], self.bound[j, 0], self.bound[j, 1])

    def creep_mutate(self):
        delta = [-1, 1]
        for i in range(self.elite_size, self.N):  # Don't mutate elite individuals
            for j in range(self.n):
                if np.random.rand() < self.mr:
                    self.population[i, j] += np.random.choice(delta, 1, replace=False)[0]
                    self.population[i, j] = np.clip(self.population[i, j], self.bound[j, 0], self.bound[j, 1])

    def adaptive_mutate_rate(self):
        # Adaptive mutation rate
        if self.gen == 0 or self.gen % self.lambda_ == 0:
            if self.diversity_method is not None:
                self.diversity, self.mr = self.mutation_adapter(self.population, self.diversity_method)

    def evolve(self):
        fitness = self.evaluate_fitness()
        self.select(fitness)
        if self.evolution_strategy == 'standard':
            self.uniform_crossover()
            self.gaussian_mutate()
        elif self.evolution_strategy == 'hux':
            self.hux_crossover()
            self.gaussian_mutate()
        elif self.evolution_strategy == 'creep_mutation':
            self.uniform_crossover()
            self.creep_mutate()
        elif self.evolution_strategy == 'hux_creep_mutation':
            self.hux_crossover()
            self.creep_mutate()
        elif self.evolution_strategy == 'adaptive_mutation':
            self.uniform_crossover()
            self.adaptive_mutate_rate()
            self.gaussian_mutate()
        elif self.evolution_strategy == 'adaptive_creep_mutation':
            self.uniform_crossover()
            self.adaptive_mutate_rate()
            self.creep_mutate()
        elif self.evolution_strategy == 'adaptive_hux':
            self.hux_crossover()
            self.adaptive_mutate_rate()
            self.gaussian_mutate()
        else:
            self.hux_crossover()
            self.adaptive_mutate_rate()
            self.creep_mutate()

        self.gen += 1
