import itertools
import math
import os
import sys

import time

import joblib
import numpy as np
import tensorflow as tf


# Get the absolute path to the directory where faires.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from faires.py
sys.path.append(os.path.join(base_path, "../../"))

from baseline.sbft.genetic_algorithm import GeneticAlgorithm as GA
from utils.helpers import get_experiment_params, generate_report
from utils.dnn_wrapper import dnn_model_wrapper


class SBFT:
    def __init__(self, config, model, classifier_name, sensitive_params, population_size=100, threshold=0):
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

        self.setup = GA(
            pop_size=population_size,
            bounds=np.array(self.config.input_bounds),
            protected_attribs=self.protected_attribs,
            fitness_func=self.discrimination_object_func(),
        )
        self.approach_name = "SBFT"

    def discrimination_object_func(self):

        def func(indv):
            """
            Sequential discrimination evaluation for a single input.
            Supports multiple protected attributes and domains.
            Behavior matches evaluate_local:
              - Stop at first discriminatory difference
              - Return 2 * abs(out1 - out0) + 1
            """

            x = np.asarray(indv, dtype=int).copy()

            # --- Clip to input bounds ---
            for j in range(self.num_attribs):
                low, high = self.input_bounds[j]
                x[j] = np.clip(x[j], int(low), int(high))

            # --- Bookkeeping ---
            self.total_generated += 1
            self.tot_inputs.add(tuple(int(v) for v in x))

            # --- All protected combinations ---
            all_combs = list(itertools.product(*self.protected_domains))

            # --- Base prediction ---
            out0 = self.model.predict(x.reshape(1, -1))[0]

            # --- Sequentially evaluate variants ---
            final_diff = 0

            for comb in all_combs:

                # Skip original combination
                is_original = True
                for j, attr_idx in enumerate(self.protected_attribs):
                    if x[attr_idx] != comb[j]:
                        is_original = False
                        break
                if is_original:
                    continue

                # Build variant
                x_var = x.copy()
                for j, attr_idx in enumerate(self.protected_attribs):
                    x_var[attr_idx] = comb[j]

                # Predict variant
                out1 = self.model.predict(x_var.reshape(1, -1))[0]

                # Compute difference
                diff = abs(out1 - out0)

                # If difference found → stop like evaluate_local
                if diff > 0 and \
                        (tuple(x) not in self.disc_inputs):
                    self.disc_inputs.add(tuple(int(v) for v in x))
                    self.disc_inputs_list.append(x.tolist())
                    self.set_time_to_1000_disc()

                    return 2 * diff + 1

                # Track last diff (in case no violation found)
                final_diff = diff

            # No discriminatory difference found
            return 2 * final_diff + 1

        return func

    def update_cumulative_efficiency(self, iteration):
        """
        Update the cumulative efficiency data if the current number of total inputs
        meets the tracking criteria (first input or every tracking_interval inputs).
        """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.disc_inputs)

        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_samples=1000 * 1000, max_allowed_time=3600):
        max_evolution = math.ceil(max_samples / self.population_size)

        for i in range(max_evolution):

            self.setup.run()
            self.update_cumulative_efficiency(i)

            use_time = time.time() - self.start_time
            if use_time >= self.log_interval:
                self.log_interval += self.initial_log_interval
                self.report(elapsed_time=use_time, is_log=True)

            if self.log_interval >= max_allowed_time or self.total_generated >= max_samples:
                break

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
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
            **additional_data,
        )

if __name__ == '__main__':
    config, sensitive_names, sensitive_params, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {",".join(map(str, sorted(sensitive_names)))}')
    print('')

    if classifier_name == 'dnn' or classifier_name == 'ftt' or classifier_name == 'tabt':
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.pkl'
        model = joblib.load(classifier_path)

    sbft = SBFT(
        config=config,
        model=model,
        sensitive_params=sensitive_params,
        classifier_name=classifier_name,
    )

    sbft.run(max_allowed_time=max_allowed_time)
