"""
ensemble_agreement.py

Computes the label agreement rate across the five ensemble members
(KNN, MLP, SVM, RF, NB) for the IDI pool sampled at each augmentation ratio.

Two metrics are reported per configuration:
  - mean_agreement   : average fraction of members agreeing with the majority
                       label across all sampled IDIs  (e.g. 0.92 = 4.6/5 on avg)
  - unanimous_rate   : fraction of IDIs where ALL five members agree (5/5)

Can be run standalone or called from the retraining script.

Standalone usage:
    cd <project_root>
    python utils/ensemble_agreement.py
    python utils/ensemble_agreement.py --fractions 0.05 0.10 0.20 1.0
    python utils/ensemble_agreement.py --approaches RAES ExpGA --classifiers rf svm
    python utils/ensemble_agreement.py --attrs multi --out results/retraining/agreement_multi.csv
"""

import argparse
import itertools
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../"))

from helpers import load_disc_inputs, get_data, get_config_dict


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_agreement(ensemble_clf, X_aug_ensemble):
    """
    Run each of the five ensemble members on X_aug_ensemble and compute
    agreement statistics.

    Returns
    -------
    mean_agreement : float
        Mean fraction of members agreeing with the majority label,
        averaged across all IDIs.  Range [0.6, 1.0] for a 5-member ensemble
        (bare majority is always at least 3/5 = 0.6).
    unanimous_rate : float
        Fraction of IDIs where all five members predict the same label.
    member_labels : ndarray of shape (n_members, N)
        Raw per-member hard predictions for downstream analysis.
    """
    voting_clf = ensemble_clf.named_steps['ensemble']
    scaler     = ensemble_clf.named_steps['scaler']
    X_scaled   = scaler.transform(X_aug_ensemble)

    # collect hard predictions from each fitted member  shape: (n_members, N)
    # estimators_ is a plain list of fitted objects; named_estimators_ gives (name, est) pairs
    member_labels = np.array([
        est.predict(X_scaled) for est in voting_clf.estimators_
    ]).astype(int)

    n_members, N = member_labels.shape

    # majority label for each instance (same logic as VotingClassifier hard vote)
    majority = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=2).argmax(), axis=0, arr=member_labels
    )

    # per-instance fraction of members agreeing with the majority
    per_instance_agreement = np.mean(member_labels == majority, axis=0)  # shape (N,)

    mean_agreement = float(per_instance_agreement.mean())
    unanimous_rate = float(np.mean(per_instance_agreement == 1.0))

    return mean_agreement, unanimous_rate, member_labels


# ---------------------------------------------------------------------------
# Sampling  — mirrors _idi_repair_dataset in retraining.py
# ---------------------------------------------------------------------------

def _sample_idi_pool(disc_inputs, full_dataset_size, aug_fraction, seed=None):
    """
    Sample IDIs at aug_fraction * full_dataset_size, matching the exact
    sampling strategy used in FairnessRetrainingRepair._idi_repair_dataset().

    Returns None if the pool is empty or the computed count is zero.
    """
    if disc_inputs is None or len(disc_inputs) == 0:
        return None

    if hasattr(disc_inputs, 'to_numpy'):
        pool = disc_inputs.to_numpy(dtype=int)
    else:
        pool = np.asarray(list(disc_inputs), dtype=int)

    num_aug = int(full_dataset_size * aug_fraction)
    num_aug = min(num_aug, len(pool))

    if num_aug == 0:
        return None

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pool), size=num_aug)
    return pool[indices]


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_agreement_analysis(approaches, fractions, classifier_names,
                           attr_mode='single', seed=42):
    """
    Loop over every combination of approach / dataset / classifier /
    attribute / augmentation fraction, compute agreement metrics, and
    return a DataFrame of results.

    Parameters
    ----------
    approaches       : list of str  e.g. ['RAES', 'ExpGA', 'SBFT', 'GRFT', 'AFT']
    fractions        : list of float  e.g. [0.05, 0.10, 0.20, 1.0]
    classifier_names : list of str  e.g. ['rf', 'svm', 'dnn']
    attr_mode        : 'single' | 'multi' | 'both'
    seed             : int — RNG seed for reproducible IDI sampling
    """
    config_dict = get_config_dict()
    records = []

    for approach_name in approaches:
        for config in config_dict.values():
            dataset_name = config.dataset_name

            # ---- load ensemble ----
            ensemble_path = os.path.join(
                'models', 'ensemble', f'{dataset_name}_ensemble.pkl'
            )
            if not os.path.exists(ensemble_path):
                print(f"[SKIP] No ensemble model found for {dataset_name}")
                continue
            ensemble_clf = joblib.load(ensemble_path)

            all_protected_indices = [idx - 1 for idx in config.sens_name.keys()]

            # ---- full dataset size (used for num_aug calculation) ----
            X_full, _, _, _ = get_data(dataset_name)()
            full_size = len(X_full)

            sensitive_combos = _build_sensitive_param_combos(config, attr_mode)

            for classifier_name in classifier_names:
                for _, sens_names_str in sensitive_combos:

                    # ---- load IDI pool ----
                    try:
                        disc_inputs = load_disc_inputs(
                            approach_name=approach_name,
                            dataset_name=dataset_name,
                            sensitive_name=sens_names_str,
                            classifier_name=classifier_name
                        )
                    except Exception as e:
                        print(f"[SKIP] {approach_name}/{dataset_name}/"
                              f"{classifier_name}/{sens_names_str}: {e}")
                        continue

                    if disc_inputs is None or len(disc_inputs) == 0:
                        print(f"[SKIP] Empty IDI pool — {approach_name}/"
                              f"{dataset_name}/{classifier_name}/{sens_names_str}")
                        continue

                    pool_size = len(disc_inputs)

                    for fraction in fractions:
                        X_aug = _sample_idi_pool(
                            disc_inputs, full_size, fraction, seed=seed
                        )
                        if X_aug is None:
                            print(f"[SKIP] Zero IDIs at fraction={fraction:.2f} — "
                                  f"{approach_name}/{dataset_name}/{classifier_name}")
                            continue

                        # drop protected attributes to match ensemble input shape
                        X_aug_ens = np.delete(X_aug, all_protected_indices, axis=1)

                        try:
                            mean_agr, unani_rate, _ = compute_agreement(
                                ensemble_clf, X_aug_ens
                            )
                        except Exception as e:
                            print(f"[ERROR] {approach_name}/{dataset_name}/"
                                  f"{classifier_name}/{sens_names_str} "
                                  f"fraction={fraction}: {e}")
                            continue

                        records.append({
                            'timestamp':       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'approach_name':   approach_name,
                            'dataset_name':    dataset_name,
                            'classifier_name': classifier_name,
                            'sensitive_name':  sens_names_str,
                            'aug_fraction':    fraction,
                            'num_aug':         len(X_aug),
                            'pool_size':       pool_size,
                            'mean_agreement':  round(mean_agr,   4),
                            'unanimous_rate':  round(unani_rate, 4),
                        })
                        print(
                            f"[OK] {approach_name:6s} | {dataset_name:7s} | "
                            f"{classifier_name:4s} | {sens_names_str:12s} | "
                            f"frac={fraction:.2f} N={len(X_aug):6d} | "
                            f"MeanAgr={mean_agr:.3f}  Unanimous={unani_rate:.3f}"
                        )

    return pd.DataFrame(records)


def _build_sensitive_param_combos(config, attr_mode='single'):
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
        names  = [config.sens_name[p] for p in params]
        result.append((params, ','.join(map(str, sorted(names)))))
    return result


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute ensemble label-agreement rates for IDI pools."
    )
    parser.add_argument(
        '--fractions', type=float, nargs='+',
        default=[0.05, 0.10, 0.20, 1.0],
        help='Augmentation fractions to evaluate (default: 0.05 0.10 0.20 1.0)'
    )
    parser.add_argument(
        '--approaches', type=str, nargs='+',
        default=['RAES', 'ExpGA', 'SBFT', 'GRFT', 'AFT'],
        help='Approaches whose IDI pools to evaluate'
    )
    parser.add_argument(
        '--classifiers', type=str, nargs='+',
        default=['rf', 'svm', 'dnn'],
        help='Classifiers to evaluate'
    )
    parser.add_argument(
        '--attrs', choices=['single', 'multi', 'both'], default='single',
        help='Attribute mode: single-attr only, multi-attr only, or both'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='RNG seed for reproducible IDI sampling (default: 42)'
    )
    parser.add_argument(
        '--out', type=str,
        default='results/retraining/ensemble_agreement.csv',
        help='Output CSV path'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Ensemble Agreement Analysis")
    print(f"  Fractions  : {args.fractions}")
    print(f"  Approaches : {args.approaches}")
    print(f"  Classifiers: {args.classifiers}")
    print(f"  Attr mode  : {args.attrs}")
    print(f"  Seed       : {args.seed}")
    print(f"  Output     : {args.out}")
    print("=" * 70)

    df = run_agreement_analysis(
        approaches=args.approaches,
        fractions=args.fractions,
        classifier_names=args.classifiers,
        attr_mode=args.attrs,
        seed=args.seed,
    )

    if df.empty:
        print("\nNo results collected — check that IDI pool files and "
              "ensemble models exist.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nFull results saved to {args.out}")

    # ---- summary table ----
    print("\n=== Mean agreement by approach × augmentation fraction ===")
    pivot = (
        df.groupby(['approach_name', 'aug_fraction'])
          [['mean_agreement', 'unanimous_rate']]
          .mean()
          .round(3)
    )
    print(pivot.to_string())

    print("\n=== Overall mean per approach ===")
    overall = (
        df.groupby('approach_name')
          [['mean_agreement', 'unanimous_rate']]
          .mean()
          .round(3)
    )
    print(overall.to_string())


if __name__ == '__main__':
    main()
