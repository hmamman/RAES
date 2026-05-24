"""
functional_redundancy.py

Measures *functional* redundancy in generated test-case pools.

Two test cases are *functionally redundant* if they are identical after the
protected attribute column(s) under test are removed — i.e. they differ only
in the protected attribute value(s) and therefore probe the same non-protected
feature combination.

Metrics reported per configuration:
  - total          : total number of test cases in the pool
  - unique_count   : distinct non-protected-feature combinations
  - func_red_rate  : (total - unique_count) / total  (0 = no redundancy)

Contrast with the *exact* redundancy rate (RR) used in the RAES paper, which
counts rows that are completely identical (including protected attributes).
Functional redundancy is a stricter measure: it reveals pools that cover few
distinct decision regions even when the protected-attribute value varies.

Standalone usage:
    cd <project_root>
    python utils/functional_redundancy.py
    python utils/functional_redundancy.py --approaches RAES ExpGA --classifiers rf svm
    python utils/functional_redundancy.py --attrs multi --out results/func_red_multi.csv
"""

import argparse
import itertools
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../"))

from helpers import load_test_inputs, get_config_dict


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_functional_redundancy(data, protected_indices_0based):
    """
    Compute functional redundancy for a test-case pool.

    Parameters
    ----------
    data : ndarray of shape (N, n_features)
        Raw test cases including all features.
    protected_indices_0based : list of int
        0-based column indices of the protected attribute(s) to drop.

    Returns
    -------
    total : int
        Number of test cases in the pool.
    unique_count : int
        Number of distinct non-protected feature combinations.
    func_red_rate : float
        (total - unique_count) / total.  0.0 means no functional redundancy.
    """
    if len(data) == 0:
        return 0, 0, 0.0

    X = np.delete(data, protected_indices_0based, axis=1)

    total = len(X)
    unique_count = len(np.unique(X, axis=0))
    func_red_rate = (total - unique_count) / total if total > 0 else 0.0

    return total, unique_count, round(func_red_rate, 6)


# ---------------------------------------------------------------------------
# Sensitive-parameter combo builder  (mirrors ensemble_agreement.py)
# ---------------------------------------------------------------------------

def _build_sensitive_param_combos(config, attr_mode='single'):
    """
    Build (protected_1based_indices, sens_names_str) tuples for a dataset config.

    attr_mode : 'single' — single-attribute only
                'multi'  — multi-attribute (intersectional) only
                'both'   — all combinations
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
        result.append((params, ','.join(map(str, sorted(names)))))
    return result


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(approaches, classifier_names, attr_mode='single'):
    """
    Iterate over every approach / dataset / classifier / sensitive-attribute
    combination, compute functional redundancy metrics, and return a DataFrame.

    Parameters
    ----------
    approaches       : list of str  e.g. ['RAES', 'ExpGA', 'SBFT', 'GRFT', 'AFT']
    classifier_names : list of str  e.g. ['rf', 'svm', 'dnn']
    attr_mode        : 'single' | 'multi' | 'both'
    """
    config_dict = get_config_dict()
    records = []

    for approach_name in approaches:
        for dataset_name, config in config_dict.items():

            sensitive_combos = _build_sensitive_param_combos(config, attr_mode)

            for classifier_name in classifier_names:
                for params_1based, sens_names_str in sensitive_combos:

                    # 0-based indices of the protected attributes under test
                    protected_idx = [p - 1 for p in params_1based]


                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {approach_name} | {dataset_name} | {classifier_name} | {sens_names_str}...")
                    try:
                        test_inputs = load_test_inputs(
                            approach_name=approach_name,
                            dataset_name=dataset_name,
                            sensitive_name=sens_names_str,
                            classifier_name=classifier_name,
                        )
                    except FileNotFoundError:
                        print(f"[SKIP] File not found — {approach_name}/"
                              f"{dataset_name}/{classifier_name}/{sens_names_str}")
                        continue
                    except Exception as e:
                        print(f"[SKIP] {approach_name}/{dataset_name}/"
                              f"{classifier_name}/{sens_names_str}: {e}")
                        continue

                    if test_inputs is None or len(test_inputs) == 0:
                        print(f"[SKIP] Empty pool — {approach_name}/"
                              f"{dataset_name}/{classifier_name}/{sens_names_str}")
                        continue

                    # We compute functional redundancy as usual.
                    total, unique_count, frr = compute_functional_redundancy(
                        test_inputs, protected_idx
                    )

                    # SPECIAL HANDLING FOR AFT:
                    # AFT saves inputs in pairs (x, x') into a set() before saving.
                    # Because it's a set, the sequential ordering is lost, so we cannot simply take [::2].
                    # However, since AFT's atomic unit of generation is a pair, its `total` row count 
                    # is exactly 2x the number of "attempts".
                    # To make its redundancy rate comparable to methods that save 1 instance per attempt,
                    # we treat its effective total as total / 2.
                    if approach_name.upper() == 'AFT':
                        total = total // 2
                        if total > 0:
                            frr = (total - unique_count) / total
                        else:
                            frr = 0.0

                    records.append({
                        'timestamp':           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'approach_name':       approach_name,
                        'dataset_name':        dataset_name,
                        'classifier_name':     classifier_name,
                        'sensitive_name':      sens_names_str,
                        'protected_indices':   str(protected_idx),
                        'total':               total,
                        'unique_count':        unique_count,
                        'func_red_rate':       frr,
                    })

                    print(
                        f"[OK] {approach_name:6s} | {dataset_name:7s} | "
                        f"{classifier_name:4s} | {sens_names_str:15s} | "
                        f"total={total:8d}  unique={unique_count:8d}  FRR={frr:.4f}"
                    )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure functional redundancy in IDI test-case pools."
    )
    parser.add_argument(
        '--approaches', type=str, nargs='+',
        default=['RAES', 'ExpGA', 'SBFT', 'GRFT', 'AFT'],
        help='Approaches whose IDI pools to evaluate'
    )
    parser.add_argument(
        '--classifiers', type=str, nargs='+',
        default=['lr', 'rf', 'svm', 'dnn'],
        help='Classifiers to evaluate'
    )
    parser.add_argument(
        '--attrs', choices=['single', 'multi', 'both'], default='single',
        help='Attribute mode: single-attr only, multi-attr only, or both'
    )
    parser.add_argument(
        '--out', type=str,
        default='results/retraining/functional_redundancy.csv',
        help='Output CSV path'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Functional Redundancy Analysis")
    print(f"  Approaches : {args.approaches}")
    print(f"  Classifiers: {args.classifiers}")
    print(f"  Attr mode  : {args.attrs}")
    print(f"  Output     : {args.out}")
    print("=" * 70)

    df = run_analysis(
        approaches=args.approaches,
        classifier_names=args.classifiers,
        attr_mode=args.attrs,
    )

    if df.empty:
        print("\nNo results collected — check that IDI pool files exist in test_data/.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nFull results saved to {args.out}")

    # ---- summary: mean FRR per approach ----
    print("\n=== Mean functional redundancy rate by approach ===")
    by_approach = (
        df.groupby('approach_name')[['total', 'unique_count', 'func_red_rate']]
          .agg({'total': 'sum', 'unique_count': 'sum', 'func_red_rate': 'mean'})
          .round({'func_red_rate': 4})
    )
    # also compute aggregate FRR from pooled totals
    by_approach['agg_frr'] = ((by_approach['total'] - by_approach['unique_count'])
                               / by_approach['total']).round(4)
    print(by_approach.to_string())

    print("\n=== Mean functional redundancy rate by approach × dataset ===")
    pivot = (
        df.groupby(['approach_name', 'dataset_name'])['func_red_rate']
          .mean()
          .round(4)
          .unstack(fill_value=float('nan'))
    )
    print(pivot.to_string())

    print("\n=== Mean functional redundancy rate by approach × classifier ===")
    pivot2 = (
        df.groupby(['approach_name', 'classifier_name'])['func_red_rate']
          .mean()
          .round(4)
          .unstack(fill_value=float('nan'))
    )
    print(pivot2.to_string())


if __name__ == '__main__':
    main()
