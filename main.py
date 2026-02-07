import argparse
import itertools
import time
import joblib
import gc
import tensorflow.keras.backend as K

# from retraining import fairness_retrain_repair
from utils.ml_classifiers import CLASSIFIERS
from utils.helpers import get_config_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument('--approach_name', type=str, default='raes', required=False, help='The name of fairness testing approach to run')
    parser.add_argument('--max_allowed_time', type=int, default=3600, help='Maximum time allowed for the experiment')
    parser.add_argument('--max_iteration', type=int, default=1, help='Maximum experiment iterations')
    return parser.parse_args()

def run_approach(approach_name, config, model, classifier_name, sensitive_params, max_time=3600):
    """Run the specified fairness testing approach."""

    if approach_name == 'raes':
        from tutorial.raes import RAES
        RAES(config=config, model=model, sensitive_params=sensitive_params, classifier_name=classifier_name, diversity_method='hamming_stratified', evolution_strategy='default', sampling_threshold=50).run(max_allowed_time=max_time)

    elif approach_name == 'expga':
        from baseline.expga.expga import ExpGA
        ExpGA(config=config, model=model, sensitive_params=sensitive_params, classifier_name=classifier_name).run(max_allowed_time=max_time)
    elif approach_name == 'sbft':
        from baseline.sbft.sbft import SBFT
        SBFT(config=config, model=model, sensitive_params=sensitive_params, classifier_name=classifier_name).run(max_allowed_time=max_time)
    elif approach_name == 'grft':
        from baseline.grft.grft import GRFT
        GRFT(config=config, model=model, sensitive_params=sensitive_params, classifier_name=classifier_name).run(max_allowed_time=max_time)

    elif approach_name == 'aft':
        from baseline.aft.aft import AFT
        AFT(config=config, model=model, sensitive_params=sensitive_params, classifier_name=classifier_name).run(max_allowed_time=max_time)


args = parse_arguments()
approaches = ['raes', 'expga', 'sbft', 'grft', 'aft']

if args.approach_name not in approaches:
    raise ValueError(f"Invalid sensitive name: {args.approach_name}. Available options are: {approaches}")

for _ in range(args.max_iteration):
    for config in get_config_dict().values():
        classifier_names = ['rf','dnn','svm']
        sensitive_params = []

        sens_list = list(config.sens_name.keys())
        # Generate combinations of length 1 and 2
        for r in list(range(1, len(sens_list) + 1)):
            sensitive_params.extend(itertools.combinations(sens_list, r))

        # Convert tuples to lists
        sensitive_params = [list(c) for c in sensitive_params]

        for classifier_name in classifier_names:
            for params in sensitive_params:
                print(f'Approach name: {args.approach_name}')
                print(f'Dataset: {config.dataset_name}')
                print(f'Classifier: {classifier_name}')
                sens_names = []
                for param in params:
                    sens_names.append(config.sens_name[param])

                print(f'Sensitive name: {",".join(map(str, sorted(sens_names)))}')

                if classifier_name == 'dnn':
                    import tensorflow as tf
                    classifier_path = f'models/{config.dataset_name}/dnn.keras'
                    model = tf.keras.models.load_model(classifier_path)
                else:
                    classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
                    model = joblib.load(classifier_path)

                run_approach(
                    approach_name=args.approach_name,
                    config=config,
                    classifier_name=classifier_name,
                    model=model,
                    sensitive_params=params,
                    max_time=args.max_allowed_time
                )
                K.clear_session()
                gc.collect()