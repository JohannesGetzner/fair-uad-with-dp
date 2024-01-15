"""This script is used to check for statistical significance"""
import os
from typing import Tuple

from scipy import stats

from src.analysis.utils import gather_data_seeds


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_metric(
        experiment_dir: str,
        attr_key: str,
        metrics: Tuple[str, str],
        alpha: float = 0.05):
    """Perform different significance test for a given metric"""

    data, _ = gather_data_seeds(experiment_dir, attr_key, metrics)

    vals1 = data[metrics[0]]  # Shape: (num_attr_key_values, num_seeds)
    vals2 = data[metrics[1]]  # Shape: (num_attr_key_values, num_seeds)

    # Perform Wilcoxon signed-rank test on balanced data
    middle_key_vals1 = vals1[vals1.shape[0] // 2]  # Shape: (num_seeds,)
    middle_key_vals2 = vals2[vals1.shape[0] // 2]  # Shape: (num_seeds,)

    """Perform tests"""
    # Two-sided test
    _, p = stats.wilcoxon(middle_key_vals1, middle_key_vals2)
    print(f'Two-sided wilcoxon signed-rank test for {metrics[0]} and {metrics[1]}: p={p}')
    if p < alpha:
        print(f'p < {alpha}: The metrics are significantly different')
    else:
        print(f'p >= {alpha}: No statistical difference')
    # Test if metric[0] is significantly larger than metric[1]
    _, p = stats.wilcoxon(middle_key_vals1, middle_key_vals2, alternative='greater')
    print(f'One-sided wilcoxon signed-rank test for {metrics[0]} > {metrics[1]}: p={p}')
    if p < alpha:
        print(f'p < {alpha}: {metrics[0]} is significantly larger than {metrics[1]}')
    else:
        print(f'p >= {alpha}: No statistical difference')
    # Test if metric[1] is significantly larger than metric[0]
    _, p = stats.wilcoxon(middle_key_vals2, middle_key_vals1, alternative='greater')
    print(f'One-sided wilcoxon signed-rank test for {metrics[1]} > {metrics[0]}: p={p}')
    if p < alpha:
        print(f'p < {alpha}: {metrics[1]} is significantly larger than {metrics[0]}')
    else:
        print(f'p >= {alpha}: No statistical difference')

    print("")


if __name__ == '__main__':
    """ FAE RSNA """
    # FAE rsna sex
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_sex')
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_AUROC", "test/lungOpacity_female_AUROC"],
        attr_key='male_percent',
    )
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_fpr@0.95", "test/lungOpacity_female_fpr@0.95"],
        attr_key='male_percent',
    )
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_anomaly_score", "test/lungOpacity_female_anomaly_score"],
        attr_key='male_percent',
    )
    # FAE rsna age
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_age')
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_AUROC", "test/lungOpacity_old_AUROC"],
        attr_key='old_percent',
    )
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_fpr@0.95", "test/lungOpacity_old_fpr@0.95"],
        attr_key='old_percent',
    )
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_anomaly_score", "test/lungOpacity_old_anomaly_score"],
        attr_key='old_percent',
    )
