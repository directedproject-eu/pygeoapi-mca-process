import json
import logging
import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from config import CONFIG
from DecisionMatrix import DecisionMatrix
from pyDOE2 import lhs

# Constants
ALT_COL = "measure"
BCR_NAME = "bcr"
NPV_NAME = "npv"
COST_KW = "cost"
N_SAMPLES_DEFAULT = 360

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(asctime)s | %(name)s::%(module)s.py:%(lineno)d | %(process)d] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_json_data(file_path: Optional[str] = None, json_str: Optional[str] = None) -> Union[Dict, List]:
    """Load JSON data from a file or a string."""
    if file_path:
        with open(file_path, "r") as file:
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
        ranks = (
            mcdm_ranks.rename(columns={"copeland": "rank"})[["measure", "rank"]].set_index("measure")["rank"].to_json()
        )

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
    input_str = os.getenv("PYGEOAPI_K8S_MANAGER_INPUTS")
    try:
        input_dict = json.loads(input_str)
    except json.JSONDecodeError as err:
        logger.error(f"Could not parse process inputs '{input_str}'. Error: '{err}'")
        raise
    except Exception as err:
        logger.error(f"Could not parse process inputs '{input_str}'. Error: '{err}'")
        raise

    weights = input_dict.get("weights")
    if weights is not None:
        VALID_KEYS = ["measure net cost", "averted risk_aai", "approval", "feasability", "durability",
                      "externalities", "implementation time"]
        if not isinstance(weights, dict):
            msg = f"'weights' has to be of type dict, but is of type '{type(weights)}'"
            logger.error(msg)
            raise TypeError(msg)
        for k, v in weights.items():
            if k not in VALID_KEYS:
                msg = f"The key '{k}' is not valid. Valid keys are: {VALID_KEYS}"
                logger.error(msg)
                raise ValueError(msg)
            if not isinstance(v, (float, int)):
                msg = f"The value '{v}' has to be of numeric type, but is of type '{type(v)}'"
                logger.error(msg)
                raise ValueError(msg)
        CONFIG["weights"] = weights
        logger.info(f"Weights were provided in the request. Config: {CONFIG}")
    else:
        logger.info(f"No weights were provided in the request, using default weights. Config: {CONFIG}")

    mode = input_dict.get("mode", "ranks")
    VALID_MODES = ["ranks", "sensitivity"]
    if mode not in VALID_MODES:
        msg = f"The mode '{mode}' is not valid. Valid modes are: {VALID_MODES}"
        logger.error(msg)
        raise ValueError(msg)

    if mode == "ranks":
        result = get_ranks(CONFIG)
    elif mode == "sensitivity":
        result = get_sensitivity(CONFIG)
    else:
        msg = "Invalid command specified in configuration."
        logger.error(msg)
        raise ValueError(msg)
    logger.info("PYGEOAPI_K8S_MANAGER_RESULT_MIMETYPE:application/json")
    process_id = os.getenv("PYGEOAPI_PROCESS_ID", "process-id-not-defined-in-env")
    logger.info(f'PYGEOAPI_K8S_MANAGER_RESULT_START\n{{"id":"{process_id}","value":{result}}}')


if __name__ == "__main__":
    main()
