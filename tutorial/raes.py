import copy
import itertools
import math
import os
import sys

import time

import joblib
import numpy as np
import tensorflow as tf

# Get the absolute path to the directory where fairamga.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairamga.py
sys.path.append(os.path.join(base_path, "../"))

from tutorial.genetic_algorithm import GA
from utils.helpers import get_experiment_params, generate_report
from utils.dnn_wrapper import dnn_model_wrapper


class RAES:
    def __init__(self, config, model, classifier_name, sensitive_params, population_size=300,
                 diversity_method='hamming_stratified', evolution_strategy='default', adaptation_scheme='linear',
                 threshold=0, sampling_threshold=50):
        self.config = config
        self.threshold = threshold
        self.population_size = population_size

        self.tot_inputs = set()
        self.disc_inputs = set()
        self.disc_inputs_list = []

        self.input_bounds = np.array(self.config.input_bounds)
        self.sensitive_params = sensitive_params
        self.model = dnn_model_wrapper(model=model)
        self.classifier_name = classifier_name

        self.protected_attribs = [sens_param - 1 for sens_param in sensitive_params]
        self.num_attribs = len(config.input_bounds)
        self.non_protected_attribs = [i for i in range(len(config.input_bounds)) if i not in self.protected_attribs]
        self.protected_domains = [list(range(int(self.input_bounds[i][0]), int(self.input_bounds[i][1]) + 1)) for i in
                                  self.protected_attribs]

        self.start_time = time.time()
        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.inference_count = 0
        self.log_interval = 300
        self.initial_log_interval = self.log_interval
        self.diversity_method = diversity_method
        self.adaptation_scheme = adaptation_scheme

        self.GA = AMGA(
            pop_size=self.population_size,
            bound=np.array(self.config.input_bounds),
            dna_size=len(self.config.input_bounds),
            discrimination_check=self.discrimination_object_func(),
            diversity_method=diversity_method,
            crossover_rate=0.8,
            mutation_rate=0.05,
            sampling_threshold=sampling_threshold,
            evolution_strategy=evolution_strategy,
        )

        self.approach_name = f"RAES"

    def similar_set_(self, X):
        """
        Generate all protected-attribute variants for a batch X.
        Returns array of shape (N*M, D), where
        - N = len(X)
        - M = number of protected combinations
        """
        X = np.array(X, dtype=int)
        N = len(X)

        all_combs = np.array(list(itertools.product(*self.protected_domains)))  # (M, K)
        M = len(all_combs)

        # Expand candidates
        batch_inputs = np.repeat(X, M, axis=0)
        for j, attr_idx in enumerate(self.protected_attribs):
            batch_inputs[:, attr_idx] = np.tile(all_combs[:, j], N)

        return batch_inputs, all_combs, M

    def discrimination_object_func(self):
        """
        Fairness objective with single-batch prediction.
        Uses similar_set_ to generate variants.
        """

        def func(indvs):
            x_array = np.asarray(indvs, dtype=int)
            N = len(x_array)
            if N == 0:
                return x_array, np.array([], dtype=int), np.array([], dtype=int)

            self.total_generated += N
            self.tot_inputs.update(
                tuple(int(v) for v in row[:self.num_attribs]) for row in x_array
            )

            # --- Generate variants ---
            batch_inputs, all_combs, M = self.similar_set_(x_array)

            orig_vals = x_array[:, self.protected_attribs]  # (N, K)
            comb_mask = (all_combs[None, :, :] == orig_vals[:, None, :]).all(axis=2)  # (N, M)
            original_indices = np.where(comb_mask.flatten())[0]

            all_outputs = self.model.predict(batch_inputs).reshape(-1)
            self.inference_count += 1

            # Recover base + variant outputs
            y_base = all_outputs[original_indices]  # (N,)
            y_matrix = all_outputs.reshape(N, M)  # (N, M)

            # # Count of variants where prediction != base
            # diff_counts = np.sum(y_matrix != y_base[:, None], axis=1)  # (N,)
            # keep_idx = np.where(diff_counts > 0)[0]  # at least one variant differs
            # x_kept = x_array[keep_idx]

            # Unique counts per candidate
            unique_counts = np.fromiter(
                (np.unique(np.concatenate(([y_base[i]], y_matrix[i]))).size for i in range(N)),
                dtype=int, count=N
            )

            # Keep discriminatory candidates
            keep_idx = np.where(unique_counts > 1)[0]
            x_kept = x_array[keep_idx]

            if len(x_kept) > 0:
                self.disc_inputs.update(
                    tuple(int(v) for v in row[:self.num_attribs]) for row in x_kept
                )
                self.set_time_to_1000_disc()

            return unique_counts

        return func

    def update_cumulative_efficiency(self, iteration, mr, diversity):
        """
        Update the cumulative efficiency data if the current number of total inputs
        meets the tracking criteria (first input or every tracking_interval inputs).
        """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc, mr, diversity])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.disc_inputs)

        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_samples=1000 * 1000, max_allowed_time=3600):
        max_evolution = math.ceil(max_samples / self.population_size)

        for i in range(max_evolution):

            self.GA.evolve()
            self.update_cumulative_efficiency(iteration=i, mr=self.GA.mr, diversity=self.GA.diversity)

            use_time = time.time() - self.start_time
            if use_time >= self.log_interval:
                self.log_interval += self.initial_log_interval
                self.report(elapsed_time=use_time, is_log=True)

            if self.log_interval >= max_allowed_time or self.total_generated >= max_samples:
                break

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        save_path = f'results'

        additional_data = {'inference_count': self.inference_count}
        sens_names = [self.config.sens_name[sens_param] for sens_param in self.sensitive_params]
        generate_report(
            approach_name=f'{self.approach_name}',
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=','.join(map(str, sorted(sens_names))),
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_inputs,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            is_log=is_log,
            save_path=save_path,
            save_data=True,
            **additional_data,
        )


if __name__ == '__main__':
    config, sensitive_names, sensitive_params, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {",".join(map(str, sorted(sensitive_names)))}')
    print('')

    if classifier_name == 'dnn':
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.pkl'
        model = joblib.load(classifier_path)

    raes = RAES(
        config=config,
        model=model,
        sensitive_params=sensitive_params,
        diversity_method='hamming_stratified',
        classifier_name=classifier_name
    )

    raes.run(max_allowed_time=max_allowed_time)
