from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from importlib.resources import files
from synthwave.utils.yaml_metadata_validator import YAMLMetadataValidator
from synthwave.synthesizer.abstract.constraints import CustomConstraint
from rdt.transformers import FloatFormatter
from sdv.constraints import create_custom_constraint_class

from synthwave.synthesizer.utils import metadata_constructor, mutator
from sdv.single_table import CTGANSynthesizer


COLUMN_NAME_MAPPING = {"indicator_sex": "Sex",
            "indicator_is_long_term_sick": "Long-term sick",
            "indicator_failed_household": "From failed household",
            'ordinal_age': 'Age',
            'ordinal_bmi_weight': 'Weight',
            "ordinal_bmi_height": "Height",
            "ordinal_education": "Education",
            "ordinal_health": "Self-perceived health",
            "ordinal_limitations_health": "Health limitations"}

CATEGORICAL_VALUE_MAPPING = {
    'Sex': {"True": "Male", "False": "Female"},
    "Long-term sick": {"True": "Yes", "False": "No"},
    "From failed household": {"True": "Yes", "False": "No"}
}

DEFAULT_TARGET_COLUMNS = ["Age", "Weight", "Height", "Education",
                   "Self-perceived health", "Health limitations"]

PERCENTILE_VALUES = [.1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9]
PERCENTILES = {f"{_v * 100:.0f}%":_v for _v in PERCENTILE_VALUES}

INTEGER_STATISTICS = ["count", "min", "max"]

def validate_column_mappings(
        name_map: dict[str, str],
        key_map: dict[str, dict[str, str]],
        targets: list[str]
) -> None:
    """Validate that all columns are properly accounted for in mappings."""
    mapped_names = set(name_map.values())
    expected_names = set(key_map.keys()).union(set(targets))

    if mapped_names != expected_names:
        missing = expected_names - mapped_names
        extra = mapped_names - expected_names
        error_msg = []
        if missing:
            error_msg.append(f"Missing mappings: {missing}")
        if extra:
            error_msg.append(f"Extra mappings: {extra}")
        raise ValueError(", ".join(error_msg))

def get_statistics_by_category(_df: pd.DataFrame,
                         _name_map: dict[str, str] | None = None,
                         _key_map: dict[str, dict[str, str]] | None = None,
                         _percentiles_dict: dict[str, float] | None = None,
                         _integer_stats: list[str] | None = None,
                         _targets: list[str] | None = None) -> pd.DataFrame:
    """
    Generate statistical summary of target columns grouped by categorical
     variables.

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe containing the data to analyze
    name_map : dict[str, str], optional
        Mapping from original column names to display names
    key_map : dict[str, dict[str, str]], optional
        Mapping for categorical values (e.g., True/False to Yes/No)
    percentiles_dict : dict[str, float], optional
        Dictionary mapping percentile labels to their values
        (e.g., "50%" -> 0.5)
    integer_stats : list[str], optional
        List of statistics that should be converted to integers
    target_columns : list[str], optional
        List of columns to compute statistics for

    Returns:
    -------
    pd.DataFrame
        Transposed dataframe with statistics for each target column by category
    """

    _name_map = _name_map or COLUMN_NAME_MAPPING
    _key_map = _key_map or CATEGORICAL_VALUE_MAPPING
    _percentiles_dict = _percentiles_dict or PERCENTILES
    _integer_stats = _integer_stats or INTEGER_STATISTICS
    _targets = _targets or DEFAULT_TARGET_COLUMNS

    validate_column_mappings(_name_map, _key_map, _targets)

    _types = ({(_c, _i): int for _c in _targets for _i in _integer_stats} |
              {(_c, _i): int for _c in _targets for _i in _percentiles_dict.keys()})

    return (_df.
     rename(columns=_name_map).
     astype({_k: str for _k in _key_map.keys()}).
     replace({_k:_v for _k, _v in _key_map.items()}).
     groupby(list(_key_map.keys()))[_targets + ["ordinal_urbanization"]].
     describe(_percentiles_dict.values()).
     astype(_types).
     drop([(x, 'count') for x in _targets[1:]] +
          [("Education", _i) for _i in _percentiles_dict.keys() if _i != "50%"] +
          [('Self-perceived health', _i) for _i in _percentiles_dict.keys()] +
          [('Health limitations', _i) for _i in _percentiles_dict.keys()],
          axis=1).T)

MAPPER = yaml.safe_load(files("synthwave.data.ehis").joinpath('wave3.yaml').read_text())

# NOTE do not replace OPCODEs with na only, there is a difference between not applicable and missing
MERGING_DATA = yaml.safe_load(files("synthwave.data.ehis").joinpath('merging.yaml').read_text())

DEFAULT_INDICATOR_MAP = {
    1: True,
    2: False
}

def pre_imputation(file_path,
                   _mapper: dict | None = None,
                   _merging_data: dict | None = None,
                   _indicator_map: dict | None = None,
                   ):

    p = Path(file_path)

    _mapper = _mapper or MAPPER
    _merging_data = _merging_data or MERGING_DATA
    _indicator_map = _indicator_map or DEFAULT_INDICATOR_MAP

    df = pd.read_csv(p,
                     delimiter=";",
                     usecols=_merging_data["KEYS"] +
                             _merging_data["TARGETS"] +
                             _merging_data["AUX"])

    _mapper = list(YAMLMetadataValidator.extract_columns(_mapper))
    _mapper = {list(i.keys())[0]:list(i.values())[0] for i in _mapper if list(i.keys())[0] in df.columns}

    df = df.rename(columns={k:YAMLMetadataValidator.build_name(v) for k, v in _mapper.items()}).replace({-1: pd.NA, -3: pd.NA})

    factor_ = 1 / (min(df["weight_person"]))

    df = df.assign(weight_person=lambda x: round(x.weight_person * factor_)).astype({'weight_person': 'uint32[pyarrow]'})

    df = df.reindex(df.index.repeat(df['weight_person'])).sample(frac=1, random_state=42).reset_index(drop=True).drop(columns='weight_person')

    df.loc[df["category_person_job_status"].isna(),["indicator_person_full_time",
                                                        "category_person_job_status2",
                                                        "category_person_job_occupation",
                                                        "category_person_job_nace"]] = pd.NA

    df["indicator_person_full_time"] = df["indicator_person_full_time"].map({1: True, 2: False, -2: False}).astype("bool[pyarrow]")
    # now, only those who have job status not 10 can have a triplet of -2 codes

    df["indicator_person_sex"] = df["indicator_person_sex"].map(_indicator_map).astype("bool[pyarrow]")
    df["indicator_person_lives_with_partner"] = df["indicator_person_lives_with_partner"].map(_indicator_map).astype("bool[pyarrow]")
    df["indicator_person_long_term_sick"] = df["indicator_person_long_term_sick"].map(_indicator_map).astype("bool[pyarrow]")

    _to_transform = [_c for _c in df.columns if _c.startswith(("category", "ordinal")) and not _c.endswith("_age")]

    df[_to_transform] = df[_to_transform].astype("int16[pyarrow]")

    df.to_parquet(p.parents[0] / 'PL_pre_imputed.parquet')

def filler(_df, _column_list):
    _unemployed = _df[_column_list[0]].ne(10) & ~_df[_column_list[1]]

    typical_values = _df[~_unemployed][_column_list[2:]].mode().values
    _df.loc[_unemployed, _column_list[2:]] = typical_values
    return _df

class UnemploymentConstraint(CustomConstraint):
    @staticmethod
    def is_valid(column_names, data):
        # category_person_job_status # 0
        # no need to check if they work full time here (and only here); this has been covered by the fixed combination constraint that is introduced at least one step before.
        # still, we keep it
        _cn = list(column_names)
        employed = data[_cn[0]].eq(10)
        unemployed = (data[_cn[0]].ne(10) &
                      ~data[_cn[1]] &
                      data[_cn[2:]].eq(-2).all(axis=1))
        return employed ^ unemployed

    @staticmethod
    def transform(column_names, data):
        """Replaces OPCODES in job-related columns with corresponding modes"""
        _cn = list(column_names)
        transformed_data = data.copy()
        transformed_data = filler(transformed_data, _cn)
        return transformed_data

    @staticmethod
    def reverse_transform(column_names, data):
        """Does the inverse transformation, pads corresponding columns with -2"""
        _cn = list(column_names)
        reversed_data = data.copy()
        reversed_data.loc[reversed_data[column_names[0]].ne(10), _cn[2:]] = -2
        return reversed_data

def build_model(df, batch_size=10_000, total_epochs=5_000, cuda=False):
    model = CTGANSynthesizer(metadata_constructor(df, "ehis"),
                             enforce_rounding=False,
                             epochs=total_epochs,
                             verbose=True,
                             cuda=cuda,
                             batch_size=batch_size
                             )

    job_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ["category_person_job_status",
                             "indicator_person_full_time"]
        }
    }
    model.add_constraints(constraints=[job_constraint])

    model.add_custom_constraint_class(
        create_custom_constraint_class(UnemploymentConstraint.is_valid,
                                       UnemploymentConstraint.transform,
                                       UnemploymentConstraint.reverse_transform),
        UnemploymentConstraint.__name__)

    model.add_constraints([UnemploymentConstraint.get_schema(
        ["category_person_job_status", "indicator_person_full_time",
         "category_person_job_status2",
         "category_person_job_occupation",
         "category_person_job_nace"])])

    model.validate(df)
    model.auto_assign_transformers(df)

    for _c, _t in model.get_transformers().items():
        if _t is not None:
            _repr = _t.computer_representation
            model.update_transformers(column_name_to_transformer={
                _c: FloatFormatter(enforce_min_max_values=True,
                                   computer_representation=_repr)})

    model.fit(df)
    return model

def populate_recipients(_recipients: pd.DataFrame,
                        model,
                        target_columns,
                        recipient_columns,
                        donor_columns,
                        age_map,
                        total_iterations=10):
    history = []
    result = []

    recipients = mutator(_recipients,
                         {"ordinal_": "uint16[pyarrow]",
                           "indicator_": "bool[pyarrow]", "category_": "int16[pyarrow]"})

    recipients['ordinal_age_band'] = pd.cut(x=recipients['ordinal_age'],
                                            bins=[15, 17, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 120],
                                            labels=['15-17', '18-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+'],
                                            include_lowest=True).map(age_map).astype("uint8[pyarrow]")
    # TODO remove hardcoded values

    recipients["ordinal_education"] = (recipients["category_education_level"] // 100).astype("uint8[pyarrow]")
    recipients["category_job_status_mapped"] = recipients["category_job_status"].mul(10).astype("uint16[pyarrow]")

    recipients["auxiliary_id_recipients"] = range(len(recipients))
    recipients = recipients[recipient_columns + ["auxiliary_id_recipients"]]

    recipients_local = recipients.copy()
    recipients_local["matched"] = False
    recipients_local["matched"] = recipients_local["matched"].astype("bool[pyarrow]")

    for _ in range(total_iterations):
        # Skip if all rows are already matched
        if recipients_local['matched'].all():
            print("All rows matched. Stopping early.")
            break

        synthetic_donors = model.sample(num_rows=10_000_000, batch_size=1_000_000)

        synthetic_donors = synthetic_donors[donor_columns + list(target_columns)]

        synthetic_donors = mutator(synthetic_donors, {"ordinal_": "uint16[pyarrow]", "indicator_": "bool[pyarrow]", "category_": "int8[pyarrow]"})

        synthetic_donors["category_person_country_birth"] = synthetic_donors["category_person_country_birth"].map({10: True, 21: False, 22: False}).astype("bool[pyarrow]")
        synthetic_donors["category_person_region"] = (synthetic_donors["category_person_region"] // 10).astype("uint8[pyarrow]")

        # Filter for unmatched rows only
        to_populate = recipients_local[~recipients_local['matched']].copy()

        # First, identify how many matching records exist for each combination of merging keys in the donor dataset
        match_counts = synthetic_donors.groupby(donor_columns).size().reset_index(name='match_count')

        # Merge this information with target dataset to know how many potential matches each target row has
        to_populate = pd.merge(
            to_populate,
            match_counts,
            how="left",
            left_on=recipient_columns,
            right_on=donor_columns
        ).drop(columns=donor_columns)

        # Assign a random index within the available range for each row
        to_populate['group_idx'] = to_populate["match_count"].map(lambda x: np.random.randint(0, max(1, x)) if x > 0 else 0).astype("uint16[pyarrow]")

        # Add cumcount index to donors
        synthetic_donors['group_idx'] = synthetic_donors.groupby(donor_columns).cumcount().astype("uint16[pyarrow]")

         # Merge using the random group_idx
        matched_in_this_round = pd.merge(
            to_populate[to_populate['match_count'] > 0],
            synthetic_donors,
            how="left",
            left_on=recipient_columns + ["group_idx"],
            right_on=donor_columns + ["group_idx"]
        ).drop(columns=donor_columns + ["group_idx", "match_count"])

        if len(matched_in_this_round) > 0:
            print(len(matched_in_this_round), f"{len(matched_in_this_round) * 100.0 / len(recipients)}%")
            # Append to result
            result.append(matched_in_this_round)

            # Update matched status in local copy
            recipients_local.loc[recipients_local['auxiliary_id_recipients'].isin(matched_in_this_round['auxiliary_id_recipients']), 'matched'] = True

        # Handle any remaining unmatched rows
        remaining_unmatched = recipients_local[~recipients_local['matched']].drop(columns=['matched'])
        print(len(remaining_unmatched))

    return pd.concat(result)
