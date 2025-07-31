import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp


def load_monthly_trip_counts(months, path_prefix):
    """
    Load and return a dictionary of trip counts per PULocationID for each month.
    """
    df_dict = {}
    for m in months:
        path = f"{path_prefix}{m}.parquet"
        df = pd.read_parquet(path)
        df_dict[f"count_trips_{m}"] = df["PULocationID"].value_counts()
    return df_dict


def run_hypothesis_tests(df_dict, reference_month="05", alpha=0.05):
    """
    Run Mann-Whitney U and KS tests against the reference month.
    Return months that are statistically similar.
    """
    similar_months = []

    ref_key = f"count_trips_{reference_month}"
    ref_values = df_dict[ref_key].values

    for key in df_dict:
        if key == ref_key:
            continue
        test_values = df_dict[key].values

        # Perform both tests
        mw = mannwhitneyu(ref_values, test_values, alternative="two-sided")
        ks = ks_2samp(ref_values, test_values)

        # Log results
        print(f"\n{ref_key} vs {key}")
        print(f"  - Mann-Whitney U p-value: {mw.pvalue:.4f}")
        print(f"  - KS test p-value: {ks.pvalue:.4f}")
        # alpha = 1 - alpha

        # If both tests have p-value > alpha, consider similar
        if mw.pvalue > alpha and ks.pvalue > alpha:
            similar_months.append(key.split("_")[-1])  # e.g., "04"

    return similar_months


def get_similar_months(
    path_prefix="data/yellow_tripdata_2022-", reference="05", alpha=0.05
):
    """
    Convenience method to return months statistically similar to the reference.
    """
    all_months = [f"{i:02d}" for i in range(1, 13)]
    df_dict = load_monthly_trip_counts(all_months, path_prefix)
    return run_hypothesis_tests(df_dict, reference_month=reference, alpha=alpha)
