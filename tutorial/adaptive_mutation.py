import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from typing import Tuple, Optional, Callable
from sklearn.cluster import KMeans


def stratified_sampling(population: np.ndarray, sampling_threshold: int, n_clusters: int = 10, **kwargs) -> np.ndarray:
    """
    Monte Carlo diversity estimation with stratified sampling.

    Simple and statistically sound: randomly sample with cluster-based stratification.
    """
    population = np.asarray(population)
    N, D = population.shape

    if N <= 1 or N <= sampling_threshold:
        return population

    # Cluster for stratification
    n_clusters_actual = min(n_clusters, N)
    kmeans = KMeans(n_clusters=n_clusters_actual).fit(population)
    labels = kmeans.labels_

    # Stratified sampling: sample proportionally from each cluster
    sampled_indices = []

    for c in range(n_clusters_actual):
        cluster_indices = np.where(labels == c)[0]
        cluster_size = len(cluster_indices)

        if cluster_size > 0:
            # Proportional allocation
            n_sample = max(1, int(sampling_threshold * cluster_size / N))
            n_sample = min(n_sample, cluster_size)

            selected = np.random.choice(cluster_indices, size=n_sample, replace=False)
            sampled_indices.extend(selected)

    # If we haven't reached threshold, add random samples
    if len(sampled_indices) < sampling_threshold:
        remaining_indices = np.setdiff1d(np.arange(N), sampled_indices)
        additional = min(sampling_threshold - len(sampled_indices), len(remaining_indices))
        sampled_indices.extend(np.random.choice(remaining_indices, size=additional, replace=False))

    sample_pop = population[sampled_indices[:sampling_threshold]]

    return sample_pop


def random_sampling(population: np.ndarray, sampling_threshold: int, **kwargs) -> np.ndarray:
    """
    Hamming distance-based diversity with random sampling for scalability.

    Formula: d_H(x,y) = (1/D) * ∑|x_i - y_i| (normalized Hamming distance)
    Diversity = (2/(k*(k-1))) * ∑∑d_H(x_i, x_j) for i<j, where x_i are sampled individuals.

    Uses random sampling for large populations to reduce pairwise comparisons
    while maintaining representativeness.

    Args:
        population: Population array of shape (N, D)
        n_samples: Number of samples to use when population > threshold
        sampling_threshold: Population size threshold to trigger sampling
        **kwargs: Additional arguments (unused)
    """
    population = np.asarray(population)
    N, D = population.shape

    if N <= 1 or N <= sampling_threshold:
        return population

    # Use random sampling for large populations
    # Sample without replacement
    sample_size = min(sampling_threshold, N)
    indices = np.random.choice(N, size=sample_size, replace=False)
    sample_pop = population[indices]

    return sample_pop

def compute_hamming_diversity(samples) -> float:
    samples = np.asarray(samples)
    N, D = samples.shape

    if N <= 1:
        return 0.0

    # Calculate pairwise normalized Hamming distances
    total_distance = 0.0
    pair_count = 0

    for i in range(N):
        for j in range(i + 1, N):
            hamming_dist = np.mean(samples[i] != samples[j])
            total_distance += hamming_dist
            pair_count += 1

    diversity = total_distance / pair_count if pair_count > 0 else 0.0

    return diversity

def compute_pairwise_diversity(samples, bounds: Optional[np.ndarray] = None) -> float:
    """
            Pairwise Euclidean distance diversity for continuous populations.

            Formula: d_ij = ||x_i - x_j||_2 / d_max
            Diversity = (2/(N*(N-1))) * ∑∑d_ij for i<j

            Reference: Morrison & De Jong (1999), "A test problem generator for non-stationary environments"
            """
    population = np.asarray(samples, dtype=float)
    N, D = population.shape

    if N <= 1:
        return 0.0

    # Calculate maximum possible distance for normalization
    if bounds is not None:
        bounds = np.asarray(bounds)
        max_distance = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
    else:
        # Use population extent as approximation
        pop_min = np.min(population, axis=0)
        pop_max = np.max(population, axis=0)
        max_distance = np.linalg.norm(pop_max - pop_min)
        if max_distance == 0:
            max_distance = 1.0

    # Calculate pairwise distances
    total_distance = 0.0
    pair_count = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(population[i] - population[j])
            total_distance += distance
            pair_count += 1
    
    # Average pairwise distance normalized by maximum possible distance
    diversity = (total_distance / pair_count) / max_distance if pair_count > 0 and max_distance > 0 else 0.0

    return diversity

class AdaptiveMutation:
    """
    Unified class for adaptive mutation strategies based on population diversity.

    Contains multiple diversity measures and mutation rate adaptation schemes.
    All diversity measures return values in [0, 1] where:
    - 0 = no diversity (all individuals identical)
    - 1 = maximum diversity
    """

    def __init__(self, mr_min: float = 0.05, mr_max: float = 0.5,
                 adaptation_scheme: str = 'linear', **adaptation_params):
        """
        Initialize adaptive mutation controller.

        Args:
            mr_min: Minimum mutation rate
            mr_max: Maximum mutation rate
            adaptation_scheme: How to map diversity to mutation rate
                             ('linear', 'exponential', 'sigmoid')
            **adaptation_params: Additional parameters for adaptation scheme
        """
        self.mr_min = mr_min
        self.mr_max = mr_max

        # Validate parameters
        if not (0 <= mr_min <= mr_max <= 1):
            raise ValueError("Must have 0 <= mr_min <= mr_max <= 1")

        # Set up adaptation scheme
        self.adaptation_scheme = adaptation_scheme
        self.adaptation_params = adaptation_params

        # Available diversity strategies
        self.strategies = {
            'hamming': self._hamming_diversity,
            'hamming_stratified': self._hamming_diversity_stratified_sampling,
            'hamming_random': self._hamming_diversity_random_sampling,
        }

    def _adapt_mutation_rate(self, diversity: float) -> float:
        """
        Convert diversity measure to mutation rate using selected adaptation scheme.

        Args:
            diversity: Diversity measure in [0, 1]

        Returns:
            Adaptive mutation rate
        """
        if self.adaptation_scheme == 'linear':
            # Linear mapping: low diversity → high mutation
            mr = self.mr_max * (1.0 - diversity)

        elif self.adaptation_scheme == 'exponential':
            # Exponential mapping: more aggressive adaptation
            alpha = self.adaptation_params.get('alpha', 2.0)
            mr = self.mr_max * ((1.0 - diversity) ** alpha)

        elif self.adaptation_scheme == 'sigmoid':
            # Sigmoid mapping: smooth transitions
            k = self.adaptation_params.get('k', 10.0)  # steepness
            x0 = self.adaptation_params.get('x0', 0.5)  # midpoint
            sigmoid = 1.0 / (1.0 + np.exp(-k * (x0 - diversity)))
            mr = self.mr_min + (self.mr_max - self.mr_min) * sigmoid

        else:
            raise ValueError(f"Unknown adaptation scheme: {self.adaptation_scheme}")

        return float(np.clip(mr, self.mr_min, self.mr_max))


    def _hamming_diversity(self, population: np.ndarray, **kwargs) -> float:
        diversity = compute_hamming_diversity(samples=population)
        return np.clip(diversity, 0.0, 1.0)

    def _hamming_diversity_stratified_sampling(self, population: np.ndarray, n_clusters: int = 10, **kwargs) -> float:
        threshold = self.adaptation_params.get('sampling_threshold')
        sample_pop = stratified_sampling(population, threshold, n_clusters=n_clusters)
        diversity = compute_hamming_diversity(samples=sample_pop)
        return np.clip(diversity, 0.0, 1.0)

    def _hamming_diversity_random_sampling(self, population: np.ndarray, **kwargs) -> float:
        threshold = self.adaptation_params.get('sampling_threshold')
        sample_pop = random_sampling(population, threshold)
        diversity = compute_hamming_diversity(samples=sample_pop)
        return np.clip(diversity, 0.0, 1.0)


    def calculate(self, population: np.ndarray, strategy: str = 'centroid', **kwargs) -> Tuple[float, float]:
        """
        Calculate diversity and adaptive mutation rate.

        Args:
            population: Population array of shape (N, D)
            strategy: Diversity strategy to use ('entropy', 'hamming', 'centroid',
                     'pairwise', 'moment', 'cv', 'phenotype')
            **kwargs: Additional parameters specific to diversity measure

        Returns:
            Tuple of (diversity, mutation_rate)
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(self.strategies.keys())}")

        # Calculate diversity using selected strategy
        diversity = self.strategies[strategy](population, **kwargs)

        # Convert to mutation rate
        mutation_rate = self._adapt_mutation_rate(diversity)

        return diversity, mutation_rate

    def __call__(self, population: np.ndarray, strategy: str = 'centroid', **kwargs) -> Tuple[float, float]:
        """Convenience method - same as calculate()."""
        return self.calculate(population, strategy, **kwargs)

    def get_available_strategies(self) -> list:
        """Return list of available diversity strategies."""
        return list(self.strategies.keys())

    def compare_strategies(self, population: np.ndarray, **kwargs) -> dict:
        """
        Compare all diversity strategies on the same population.

        Returns:
            Dictionary with strategy names as keys and (diversity, mutation_rate) tuples as values
        """
        results = {}
        for strategy_name in self.strategies:
            try:
                results[strategy_name] = self.calculate(population, strategy_name, **kwargs)
            except Exception as e:
                results[strategy_name] = (0.0, self.mr_max)  # fallback
                print(f"Warning: Strategy '{strategy_name}' failed: {e}")

        return results

