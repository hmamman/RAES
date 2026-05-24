import copy
import itertools
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Get the absolute path to the directory where this file is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up
sys.path.append(os.path.join(base_path, "../"))

from helpers import get_experiment_params, load_disc_inputs, get_data, get_config_dict
from dnn_wrapper import dnn_model_wrapper as model_wrapper


class FairnessRetrainingRepair:
    def __init__(self, approach_name, config, model, classifier_name, disc_inputs, X, Y, sensitive_params,
                 sensitive_names, threshold=0.5, retrain_fraction=0.10,
                 num_aug_override=None, max_aug_fraction=0.05,
                 eval_random_size=10000, eval_chunk_size=512):
        self.binary_threshold = threshold
        self.approach_name = approach_name
        self.config = config
        self.input_bounds = config.input_bounds
        # self.threshold = threshold
        self.retrain_fraction = retrain_fraction
        self.num_aug_override = num_aug_override  # Fixed count for equal-set mode
        self.max_aug_fraction = max_aug_fraction  # Cap for equal-set (default 5% of training)
        self.eval_random_size = eval_random_size
        self.eval_chunk_size = eval_chunk_size

        self.sensitive_params = sensitive_params
        self.sensitive_names = sensitive_names
        self.protected_attribs = [param-1 for param in sensitive_params]  # 0-based
        self.num_attribs = len(config.input_bounds)

        self._orig_model = model
        self._orig_model_copy = copy.deepcopy(model)  # Preserve pristine copy
        self.model = model_wrapper(model=model)
        self.classifier_name = classifier_name

        # NOTE: disc_inputs is expected to be an iterable/set of tuples (feature vector)
        self.disc_inputs = disc_inputs

        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

    # --------------------------
    # Helpers
    # --------------------------
    def _predict_raw(self, X):
        return self.model.predict(np.asarray(X))

    def _as_labels(self, y_raw):
        y = np.asarray(y_raw)
        if y.ndim == 1:
            # numeric vector -> threshold to 0/1
            return (y >= self.binary_threshold).astype(int)
        elif y.ndim == 2:
            # probabilities/logits -> argmax
            return np.argmax(y, axis=1)
        else:
            raise ValueError("Unexpected prediction shape.")

    def predict_labels(self, X):
        return self._as_labels(self._predict_raw(X))

    def similar_set_(self, X):
        """
        Generate all protected-attribute variants for a batch X.
        Returns (batch_inputs, all_combs, M).
        """
        X = np.array(X, dtype=int)
        N = len(X)
        protected_domains = [
            list(range(int(self.input_bounds[i][0]), int(self.input_bounds[i][1]) + 1))
            for i in self.protected_attribs
        ]
        all_combs = np.array(list(itertools.product(*protected_domains)), dtype=int)  # (M, K)
        M = len(all_combs)

        batch_inputs = np.repeat(X, M, axis=0)
        for j, attr_idx in enumerate(self.protected_attribs):
            batch_inputs[:, attr_idx] = np.tile(all_combs[:, j], N)

        return batch_inputs, all_combs, M

    def is_idi_mask(self, X_batch):
        """
        True for rows in X_batch that are IDIs (different outputs across protected variants).
        """
        X_batch = np.asarray(X_batch, dtype=int)
        N = len(X_batch)
        batch_inputs, all_combs, M = self.similar_set_(X_batch)

        # index of the original variant for each row
        orig_vals = X_batch[:, self.protected_attribs]  # (N, K)
        comb_mask = (all_combs[None, :, :] == orig_vals[:, None, :]).all(axis=2)  # (N, M)
        original_indices = np.where(comb_mask.flatten())[0]

        all_labels = self.predict_labels(batch_inputs)
        y_base = all_labels[original_indices]     # (N,)
        y_matrix = all_labels.reshape(N, M)       # (N, M)

        # unique_counts = np.fromiter(
        #     (np.unique(np.concatenate([[y_base[i]], y_matrix[i]])).size for i in range(N)),
        #     dtype=int, count=N
        # )
        # return unique_counts > 1
        diff_counts = np.sum(y_matrix != y_base[:, None], axis=1)
        return diff_counts > 0

    def chunk(self, data, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    # --------------------------
    # Metrics: Acc, IFo, IFr
    # --------------------------
    def accuracy(self, model_labels, true_labels):
        return float((np.asarray(model_labels) == np.asarray(true_labels)).mean())

    def compute_all_metrics(self, model_labels, true_labels):
        acc = self.accuracy(model_labels, true_labels)
        if len(np.unique(true_labels)) > 2:
            prec = precision_score(true_labels, model_labels, average='macro', zero_division=0)
            rec = recall_score(true_labels, model_labels, average='macro', zero_division=0)
            f1 = f1_score(true_labels, model_labels, average='macro', zero_division=0)
        else:
            prec = precision_score(true_labels, model_labels, zero_division=0)
            rec = recall_score(true_labels, model_labels, zero_division=0)
            f1 = f1_score(true_labels, model_labels, zero_division=0)
        return acc, prec, rec, f1

    def _ratio_idi(self, X_eval):
        total = len(X_eval)
        if total == 0:
            return 0.0
        count = 0
        for c in self.chunk(X_eval, self.eval_chunk_size):
            mask = self.is_idi_mask(c)
            count += int(mask.sum())
        return count / total

    def compute_IFo(self):
        # IDI ratio on the original dataset
        return self._ratio_idi(self.X)

    def compute_IFr(self):
        # IDI ratio on a random set sampled from input bounds
        D = self.num_attribs
        lows = np.array([int(self.input_bounds[i][0]) for i in range(D)])
        highs = np.array([int(self.input_bounds[i][1]) for i in range(D)])
        Xr = np.vstack([np.random.randint(lows, highs + 1) for _ in range(self.eval_random_size)])
        return self._ratio_idi(Xr)

    # --------------------------
    # Repair set from existing IDIs
    # --------------------------
    def _majority_label_over_similar_set(self, x):
        """
        Assign one label to all protected variants of x by majority of model predictions.
        """
        batch_inputs, _, _ = self.similar_set_(x[None, :])
        labels = self.predict_labels(batch_inputs)
        vals, counts = np.unique(labels, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _idi_repair_dataset(self):
        """
        Randomly samples raw IDIs generated uniformly using an RNG integer pull.
        Label assignment: Drops the protected attribute columns temporarily and assigns
        labels to the IDIs based on the prediction of an external, pre-loaded ensemble_clf
        """
        if self.disc_inputs is None or len(self.disc_inputs) == 0:
            return np.empty((0, self.num_attribs), dtype=int), np.empty((0,), dtype=int)

        if hasattr(self.disc_inputs, 'to_numpy'):
            disc_pool = self.disc_inputs.to_numpy(dtype=int)
        else:
            disc_pool = np.asarray(list(self.disc_inputs), dtype=int)

        N = len(disc_pool)
        train_size = len(self.X)

        if self.num_aug_override is not None:
            # Equal-set mode: use fixed count, capped at max_aug_fraction of training data
            max_allowed = int(train_size * self.max_aug_fraction)
            num_aug = min(self.num_aug_override, max_allowed, N)
        else:
            # Fraction mode: fraction is relative to training data size
            num_aug = int(train_size * self.retrain_fraction)
            num_aug = min(num_aug, N)  # Cap at available IDIs

        if num_aug == 0:
            return np.empty((0, self.num_attribs), dtype=int), np.empty((0,), dtype=int)

        # Randomly sample num_aug IDIs (with replacement, as in baseline snippet)
        rand_indices = np.random.randint(0, N, size=num_aug)
        X_aug = disc_pool[rand_indices]

        # Process ensemble prediction
        # The ensemble models are now trained without ANY protected features (based on config.sens_name)
        ensemble_path = os.path.join('models', 'ensemble', f'{self.config.dataset_name}_ensemble.pkl')
        
        try:
            ensemble_clf = joblib.load(ensemble_path)
            
            # To match the ensemble's training shape, we must delete ALL protected attributes
            # defined in the configuration, not just the one currently being tested.
            all_protected_indices = [idx - 1 for idx in self.config.sens_name.keys()]
            X_aug_ensemble = np.delete(X_aug, all_protected_indices, axis=1)
            
            # Predict using the unaware ensemble
            y_aug_raw = ensemble_clf.predict(X_aug_ensemble)
            
            # Use _as_labels to ensure binary labeling thresholding
            Y_aug = self._as_labels(y_aug_raw)
            
        except Exception as e:
            # Fallback logic if the ensemble is missing or the feature count still mismatches (backwards compatibility)
            print(f"[{self.approach_name}] Could not use ensemble for labeling {self.config.dataset_name}: {e}. Fallback to majority rule.")
            
            Y_aug_list = []
            for x in X_aug:
                # Majority vote fallback uses the model under test's own predictions across protected variants
                Y_aug_list.append(self._majority_label_over_similar_set(x))
            Y_aug = np.array(Y_aug_list)

        return X_aug, Y_aug

    # --------------------------
    # Retraining
    # --------------------------
    def _is_tf_keras(self):
        try:
            import tensorflow as tf  # noqa: F401
        except Exception:
            return False
        return hasattr(self._orig_model, "fit") and hasattr(self._orig_model, "save")

    def _num_classes(self):
        return int(len(np.unique(self.Y)))

    def retrain_with_idis(self):
        """
        Retrain CuT with IDIs sampled and labeled via ensemble predict.
        """
        X_idi, Y_idi = self._idi_repair_dataset()

        if len(X_idi) == 0:
            print(f"[{self.approach_name}] No IDIs available; skip retraining.")
            return

        X_re = np.vstack([self.X, X_idi])
        Y_re = np.concatenate([self.Y, Y_idi])

        X_train, X_val, y_train, y_val = train_test_split(X_re, Y_re, test_size=0.2, random_state=42)

        if self._is_tf_keras():
            import tensorflow as tf
            from tensorflow import keras
            model = self._orig_model
            num_classes = self._num_classes()

            # Always recompile to avoid TypeError with steps_per_execution and corrupted optimizer load
            loss = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=["accuracy"])
            
            # Explicitly configure to avoid NoneType integer comparison in Keras
            if hasattr(model, 'steps_per_execution') and model.steps_per_execution is None:
                model.steps_per_execution = 1

            model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)], batch_size=128, verbose=0)

            out_path = f"models/retrained/{self.approach_name}/{self.config.dataset_name}/{self.classifier_name}_{self.sensitive_names}_retrained.keras"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            model.save(out_path)
            print(f"[{self.approach_name}] Saved retrained Keras model to {out_path}")
        else:
            # sklearn-style
            if not hasattr(self._orig_model, "fit"):
                raise RuntimeError("Model is not retrainable via .fit().")
            model = self._orig_model
            model.fit(X_re, Y_re) # test split is primarily for NN, fallback to standard fit for sklearn
            out_path = f"models/retrained/{self.approach_name}/{self.config.dataset_name}/{self.classifier_name}_{self.sensitive_names}_retrained.pkl"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            joblib.dump(model, out_path)
            print(f"[{self.approach_name}] Saved retrained sklearn model to {out_path}")

        # refresh wrapper with the retrained model
        self.model = model_wrapper(model=self._orig_model)

    def reset_model(self):
        """Restore model to its pristine pre-retraining state."""
        self._orig_model = copy.deepcopy(self._orig_model_copy)
        self.model = model_wrapper(model=self._orig_model)

    # --------------------------
    # Main
    # --------------------------
    def run(self):
        # Mirror the 80/20 split used in build_classifiers.py so accuracy is
        # measured on the same held-out test set, not on training data.
        _, X_test, _, Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42
        )

        # BEFORE
        y_pred_before = self.predict_labels(X_test)
        acc_before, prec_before, rec_before, f1_before = self.compute_all_metrics(y_pred_before, Y_test)
        ifo_before = self.compute_IFo()
        ifr_before = self.compute_IFr()
        print(f"[{self.approach_name}] BEFORE  | Acc={acc_before:.3f} Prec={prec_before:.3f} Rec={rec_before:.3f} F1={f1_before:.3f} IFo={ifo_before:.3f}  IFr={ifr_before:.3f}")

        # RETRAIN using existing IDIs
        self.retrain_with_idis()

        # AFTER
        y_pred_after = self.predict_labels(X_test)
        acc_after, prec_after, rec_after, f1_after = self.compute_all_metrics(y_pred_after, Y_test)
        ifo_after = self.compute_IFo()
        ifr_after = self.compute_IFr()
        print(f"[{self.approach_name}] AFTER   | Acc={acc_after:.3f} Prec={prec_after:.3f} Rec={rec_after:.3f} F1={f1_after:.3f} IFo={ifo_after:.3f}  IFr={ifr_after:.3f}")

        # DELTAS
        print(f"[{self.approach_name}] ΔAcc={acc_after - acc_before:+.3f}  "
              f"ΔIFo={ifo_after - ifo_before:+.3f}  "
              f"ΔIFr={ifr_after - ifr_before:+.3f}")

        save_retraining_results(
            approach_name=self.approach_name,
            dataset_name=self.config.dataset_name,
            sensitive_name=self.sensitive_names,
            classifier_name=self.classifier_name,
            retrain_fraction=self.retrain_fraction,
            num_aug_override=self.num_aug_override,
            acc_before=acc_before,
            prec_before=prec_before,
            rec_before=rec_before,
            f1_before=f1_before,
            ifo_before=ifo_before,
            ifr_before=ifr_before,
            acc_after=acc_after,
            prec_after=prec_after,
            rec_after=rec_after,
            f1_after=f1_after,
            ifo_after=ifo_after,
            ifr_after=ifr_after
        )


def save_retraining_results(approach_name, dataset_name, sensitive_name, classifier_name,
                     retrain_fraction, num_aug_override,
                     acc_before, prec_before, rec_before, f1_before, ifo_before, ifr_before,
                     acc_after, prec_after, rec_after, f1_after, ifo_after, ifr_after):
    """
    Save AFT retraining results to CSV, appending across multiple runs.
    Includes timestamp for traceability.
    """

    # Build dictionary of results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "approach_name": approach_name,
        "dataset_name": dataset_name,
        "sensitive_name": sensitive_name,
        "classifier_name": classifier_name,
        "retrain_fraction": retrain_fraction,
        "num_aug_override": num_aug_override,
        "acc_before": round(acc_before, 3),
        "prec_before": round(prec_before, 3),
        "rec_before": round(rec_before, 3),
        "f1_before": round(f1_before, 3),
        "ifo_before": round(ifo_before, 3),
        "ifr_before": round(ifr_before, 3),
        "acc_after": round(acc_after, 3),
        "prec_after": round(prec_after, 3),
        "rec_after": round(rec_after, 3),
        "f1_after": round(f1_after, 3),
        "ifo_after": round(ifo_after, 3),
        "ifr_after": round(ifr_after, 3),
        "delta_acc": round(acc_after - acc_before, 3),
        "delta_ifo": round(ifo_after - ifo_before, 3),
        "delta_ifr": round(ifr_after - ifr_before, 3),
    }

    path = f"results/retraining/{approach_name}_retrained.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Wrap in DataFrame for saving
    df = pd.DataFrame([results])

    if os.path.isfile(path):
        # Load existing CSV
        existing_df = pd.read_csv(path)

        # Ensure schema alignment (add missing cols if needed)
        missing_columns = set(df.columns) - set(existing_df.columns)
        if missing_columns:
            for col in missing_columns:
                existing_df[col] = None
            existing_df.to_csv(path, index=False)

        # Append new row without header
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        # Create file with header
        df.to_csv(path, index=False)

    print(f"[{approach_name}] Results saved to {path}")

def _load_model(config, classifier_name):
    """Load a fresh model from disk."""
    if classifier_name == 'dnn':
        import tensorflow as tf
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.keras'
        return tf.keras.models.load_model(classifier_path)
    else:
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.pkl'
        return joblib.load(classifier_path)


def fairness_retrain_repair(approach_name, config, classifier_name, sensitive_params,
                           retrain_fraction=0.10, num_aug_override=None):
    model = _load_model(config, classifier_name)

    sens_names_list = [config.sens_name[sens_param] for sens_param in sensitive_params]
    sensitive_names = ','.join(map(str, sorted(sens_names_list)))

    disc_inputs = load_disc_inputs(
        approach_name=approach_name,
        dataset_name=config.dataset_name,
        sensitive_name=sensitive_names,
        classifier_name=classifier_name
    )

    X, Y, _, _ = get_data(config.dataset_name)()

    retrain_repair = FairnessRetrainingRepair(
        approach_name=approach_name,
        config=config,
        model=model,
        classifier_name=classifier_name,
        disc_inputs=disc_inputs,
        X=X,
        Y=Y,
        sensitive_params=sensitive_params,
        sensitive_names=sensitive_names,
        retrain_fraction=retrain_fraction,
        num_aug_override=num_aug_override
    )

    retrain_repair.run()


def compute_min_idi_counts(approaches, config_dict, classifier_names):
    """
    For each (dataset, classifier, attribute_combo) config, compute the minimum #IDI
    across all approaches. Returns dict keyed by (dataset, classifier, sensitive_names_str).
    """
    min_counts = {}
    for config in config_dict.values():
        sensitive_param_combos = _build_sensitive_param_combos(config)
        for classifier_name in classifier_names:
            for params, sens_names_str in sensitive_param_combos:
                key = (config.dataset_name, classifier_name, sens_names_str)
                counts = []
                for approach in approaches:
                    try:
                        disc = load_disc_inputs(approach, config.dataset_name, sens_names_str, classifier_name)
                        counts.append(len(disc))
                    except Exception:
                        pass  # Approach may not have data for this config
                if counts:
                    min_counts[key] = min(counts)
    return min_counts

def _build_sensitive_param_combos(config, attr_mode='single'):
    """
    Build attribute combinations from config.sens_name.
    attr_mode: 'single' (length-1 only), 'multi' (length>1 only), 'both' (all).
    Returns list of (params_list, sensitive_names_str) tuples.
    """
    sens_list = list(config.sens_name.keys())
    combos = []
    for r in range(1, len(sens_list) + 1):
        combos.extend(itertools.combinations(sens_list, r))

    result = []
    for combo in combos:
        if attr_mode == 'single' and len(combo) > 1:
            continue
        if attr_mode == 'multi' and len(combo) == 1:
            continue
        params = list(combo)
        names = [config.sens_name[p] for p in params]
        names_str = ','.join(map(str, sorted(names)))
        result.append((params, names_str))
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Fairness retraining experiments")
    parser.add_argument('--mode', choices=['sensitivity', 'equal_set', 'default', 'agreement'],
                        default='default',
                        help='Experiment mode: sensitivity, equal_set, default, or '
                             'agreement (ensemble label-agreement analysis only)')
    parser.add_argument('--repeats', type=int, default=10, help='Number of repeats')
    parser.add_argument('--attrs', choices=['single', 'multi', 'both'], default='both',
                        help='Attribute mode: single-attr only, multi-attr only, or both')
    parser.add_argument('--fractions', type=float, nargs='+',
                        default=[0.05, 0.10, 0.20, 1.0],
                        help='Augmentation fractions for agreement mode '
                             '(default: 0.05 0.10 0.20 1.0)')
    args = parser.parse_args()

    my_approaches = ['RAES', 'ExpGA', 'SBFT', 'GRFT', 'AFT']
    my_classifier_order = ['lr', 'rf', 'svm', 'dnn']

    if args.mode == 'agreement':
        from ensemble_agreement import run_agreement_analysis
        df = run_agreement_analysis(
            approaches=my_approaches,
            fractions=args.fractions,
            classifier_names=my_classifier_order,
            attr_mode=args.attrs,
        )
        if not df.empty:
            out = 'results/retraining/ensemble_agreement.csv'
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df.to_csv(out, index=False)
            print(f"\nAgreement results saved to {out}")
            print("\n=== Mean agreement by approach × fraction ===")
            print(df.groupby(['approach_name', 'aug_fraction'])
                    [['mean_agreement', 'unanimous_rate']].mean().round(3).to_string())

    elif args.mode == 'sensitivity':
        # Item 6.1: Augmentation sensitivity at 5%, 10%, 20%
        fractions = [0.10, 0.20, 1.0]
        # fractions = [0.05]
        for repeat in range(args.repeats):
            for fraction in fractions:
                for approach_name in my_approaches:
                    for config in get_config_dict().values():
                        sensitive_param_combos = _build_sensitive_param_combos(config, args.attrs)
                        for classifier_name in my_classifier_order:
                            for params, sens_names_str in sensitive_param_combos:
                                print(f'[Sensitivity] Repeat={repeat+1} Fraction={fraction} '
                                      f'Approach={approach_name} Dataset={config.dataset_name} '
                                      f'Classifier={classifier_name} Attr={sens_names_str}')
                                try:
                                    fairness_retrain_repair(
                                        approach_name, config, classifier_name,
                                        params, retrain_fraction=fraction
                                    )
                                except Exception as e:
                                    print(f'  ERROR: {e}')

    elif args.mode == 'equal_set':
        # Item 6.2: Controlled equal-set retraining
        print('Computing minimum IDI counts across approaches...')
        min_counts = compute_min_idi_counts(my_approaches, get_config_dict(), my_classifier_order)
        print(f'Computed min counts for {len(min_counts)} configurations.')

        for repeat in range(args.repeats):
            for approach_name in my_approaches:
                for config in get_config_dict().values():
                    sensitive_param_combos = _build_sensitive_param_combos(config, args.attrs)
                    for classifier_name in my_classifier_order:
                        for params, sens_names_str in sensitive_param_combos:
                            key = (config.dataset_name, classifier_name, sens_names_str)
                            if key not in min_counts:
                                continue
                            num_aug = min_counts[key]
                            print(f'[EqualSet] Repeat={repeat+1} N={num_aug} '
                                  f'Approach={approach_name} Dataset={config.dataset_name} '
                                  f'Classifier={classifier_name} Attr={sens_names_str}')
                            try:
                                fairness_retrain_repair(
                                    approach_name, config, classifier_name,
                                    params, num_aug_override=num_aug
                                )
                            except Exception as e:
                                print(f'  ERROR: {e}')

    else:
        # Default: original 10% retraining
        for repeat in range(args.repeats):
            for approach_name in my_approaches:
                for config in get_config_dict().values():
                    sensitive_param_combos = _build_sensitive_param_combos(config, args.attrs)
                    for classifier_name in my_classifier_order:
                        for params, sens_names_str in sensitive_param_combos:
                            print(f'Approach={approach_name} Dataset={config.dataset_name} '
                                  f'Classifier={classifier_name} Attr={sens_names_str}')
                            try:
                                fairness_retrain_repair(
                                    approach_name, config, classifier_name, params
                                )
                            except Exception as e:
                                print(f'  ERROR: {e}')
