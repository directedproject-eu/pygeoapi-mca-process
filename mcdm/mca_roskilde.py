import json
import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pyDOE2 import lhs

# Assuming DecisionMatrix is a class you have defined elsewhere
from DecisionMatrix import DecisionMatrix

# Constants
ALT_COL = "measure"
BCR_NAME = "bcr"
NPV_NAME = "npv"
COST_KW = "cost"
N_SAMPLES_DEFAULT = 360

# Set up logging
logger = logging.getLogger(__name__)


def load_json_data(file_path: Optional[str] = None, json_str: Optional[str] = None) -> Union[Dict, List]:
    """Load JSON data from a file or a string."""
    if file_path:
        with open(file_path, 'r') as file:
            return json.load(file)
    elif json_str:
        return json.loads(json_str)
    else:
        raise ValueError("Either file_path or json_str must be provided.")


def add_criteria(df: pd.DataFrame, criteria: Dict) -> pd.DataFrame:
    """Add criteria to the DataFrame based on the mapping dictionary."""
    for criterion_name, mapping_dict in criteria.items():
        df[criterion_name] = df[ALT_COL].map(mapping_dict)
    return df


def get_ranks(config: Dict) -> str:
    """Calculate ranks based on the provided configuration."""
    try:
        metrics = config["metrics"]
        criterias_to_consider = config["criterias_to_consider"]
        custom_criterias = config.get("custom_criterias", {})
        weights = config.get("weights", {})
        constraints = config.get("constraints", {})
        group_cols = config.get("group_cols", [])

        decision_matrix = pd.DataFrame(metrics)
        decision_matrix = add_criteria(decision_matrix, custom_criterias)

        all_crit_cols = [col for col in decision_matrix.columns if col in criterias_to_consider]
        alt_cols = [ALT_COL]
        crit_cols = [col for col in all_crit_cols if BCR_NAME not in col and NPV_NAME not in col]
        objectives = {crit: -1 if COST_KW in crit else 1 for crit in crit_cols}

        dm = DecisionMatrix(
            metrics_df=decision_matrix,
            objectives=objectives,
            alt_cols=alt_cols,
            crit_cols=crit_cols,
            group_cols=group_cols,
            weights=weights,
        )

        dm_groups = dm.pivot_and_reweight_criteria(piv_col="group")
        mcdm_ranks = dm_groups.calc_rankings(constraints=constraints).ranks_MCDM_df
        ranks = mcdm_ranks.rename(columns={"copeland": "rank"})[["measure", "rank"]].set_index("measure")["rank"].to_json()

        return ranks
    except Exception as e:
        logger.error(f"Error calculating ranks: {e}")
        raise


def get_sensitivity(config: Dict) -> str:
    """Perform sensitivity analysis based on the provided configuration."""
    try:
        metrics = config.get("metrics")
        criterias_to_consider = config.get("criterias_to_consider")
        custom_criterias = config.get("custom_criterias", {})
        weights = config.get("weights", {})
        constraints = config.get("constraints", {})
        group_cols = config.get("group_cols", [])
        n_samples = config.get("n_samples", 360)

        decision_matrix = pd.DataFrame(metrics)
        decision_matrix = add_criteria(decision_matrix, custom_criterias)

        all_crit_cols = [col for col in decision_matrix.columns if col in criterias_to_consider]
        alt_cols = [ALT_COL]
        crit_cols = [col for col in all_crit_cols if BCR_NAME not in col and NPV_NAME not in col]
        objectives = {crit: -1 if COST_KW in crit else 1 for crit in crit_cols}

        weight_samples = lhs(len(criterias_to_consider), samples=n_samples, random_state=np.random.RandomState(seed=1))
        all_ranks = []

        for i in range(n_samples):
            weights = {k: v for k, v in zip(weights.keys(), weight_samples[i])}
            dm = DecisionMatrix(
                metrics_df=decision_matrix,
                objectives=objectives,
                alt_cols=alt_cols,
                crit_cols=crit_cols,
                group_cols=group_cols,
                weights=weights,
            )
            dm_groups = dm.pivot_and_reweight_criteria(piv_col="group")
            mcdm_ranks = dm_groups.calc_rankings(constraints=constraints).ranks_MCDM_df
            ranks = mcdm_ranks.rename(columns={"copeland": "rank"})[["measure", "rank"]]
            all_ranks.append(ranks)

        all_ranks = pd.concat(all_ranks)
        sensitivity = all_ranks.groupby("measure")["rank"].value_counts() / all_ranks.groupby("measure")["rank"].count()
        sensitivity = sensitivity.unstack(0).fillna(0)

        return sensitivity.to_json()
    except Exception as e:
        logger.error(f"Error performing sensitivity analysis: {e}")
        raise


def main():
    logger.info("Start running MCA Roskilde process.")
    config = os.getenv("CONFIG")
    mode = os.getenv("MODE", "ranks")

    config = load_json_data(json_str=config) if config.startswith("{") else load_json_data(file_path=config)

    if mode == 'ranks':
        ranks = get_ranks(config)
        print(ranks)
    elif mode == 'sensitivity':
        sensitivity = get_sensitivity(config)
        print(sensitivity)
    else:
        raise ValueError("Invalid command specified in configuration.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
