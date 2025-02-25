from collections import Counter

import pandas as pd
import yaml
from sdv.metadata import Metadata

from importlib.resources import files
METADATA = yaml.safe_load(files("synthwave.data.understanding_society").joinpath('meta_metadata.yaml').read_text())

def metadata_constructor(df: pd.DataFrame, _table_name: str) -> Metadata:
    """
    It is assumed at this stage that there is no NA, NaN in the table;
    Also, degenerate columns and ordinals with two categories only
    have been removed or reduced, respectively.

    Parameters
    ----------
    _table_name
    df

    Returns
    -------

    """

    # no floats any more
    # this could be implemented better,
    # but we'll keep it that way to make sure no extra columns get in
    md = Metadata()
    md.add_table(_table_name)

    for _c in df.columns:
        if (prefix := _c.split("_")[0] + "_") in METADATA.keys():
            if METADATA[prefix]['sdtype'] == 'numerical':
                if prefix in ["ordinal_", "income_"]:
                    if (_min := df[_c].min()) >= 0:
                        match _max := df[_c].max():
                            case _ if _max <= 255:
                                _t = 'UInt8'
                            case _ if _max <= 65_535:
                                _t = 'UInt16'
                            case _ if _max <= 4_294_967_295:
                                _t = 'UInt32'
                            case _:
                                _t = 'UInt64'
                    else:
                        match _max := df[_c].max():
                            case _ if _max <= 127:
                                _t = 'Int8'
                            case _ if _max <= 32_767:
                                _t = 'Int16'
                            case _ if _max <= 2_147_483_647:
                                _t = 'Int32'
                            case _:
                                _t = 'Int64'
                    md.add_column(column_name=_c,
                                  sdtype=METADATA[prefix]["sdtype"],
                                  table_name=_table_name,
                                  computer_representation=_t)
                else:
                    md.add_column(column_name=_c,
                                  sdtype=METADATA[prefix]["sdtype"],
                                  table_name=_table_name,
                                  computer_representation=METADATA[prefix]["computer_representation"])
            else:
                md.add_column(column_name=_c,
                              sdtype=METADATA[prefix]["sdtype"],
                              table_name=_table_name)
        else:
            raise TypeError(f"No valid md template provided for column {_c}")
    md.validate()
    return md

def vote(_values):
    _targets = []
    for _v in _values:
        if not pd.isna(_v):
            _targets.append(_v)

    match len(_targets):
        case 0: return pd.NA
        case 1: return _targets[0]
        case _: return Counter(_targets).most_common(1)[0][0]
# TODO at this stage every cloned hh should be seen as unique but here we pile them all together

def clean_imputed_data(_df: pd.DataFrame = None) -> pd.DataFrame:

    redundant_ids = []
    for _c in _df.columns:
        if _c.startswith("id_") and _c != "id_household":
            redundant_ids.append(_c)
    redundant_ids.remove("id_person")
    redundant_ids.remove("id_partner")
    _df = _df.drop(columns=redundant_ids)

    _df = _df.drop(columns=["Unnamed: 0", ".imp", ".id"])

    _df = _df.drop(columns=["ordinal_household_year", # year was needed for income imputation mostly
                              "has_partner", # this one is inferred from other fields and was needed for imputation; don't need for singles and couples, m2 are not partners, we have to ignore this for m3+ and mf who might have partners
                              "total_individuals", # total_individuals is not needed here too, it was used for classification only; it can be recovered later if needed
                              # total_children should stay for household synthesis
                              "category_person_final_computed_outcome",
                              "category_person_individual_interview_outcome",
                              "category_person_self_completion"])
    # NOTE we only used these columns for imputation purposes; these are mandatory to drop for everyone. This is applied to all individual records

    _hh = []
    for _c in _df.columns:
        if "household" in _c and not (_c.startswith("id_") or (_c == "category_household_type") or (_c == "category_household_location")):
            _hh.append(_c)

    # At this stage there MUST be no empty records - aside from ids - therefore any NA checks are redundant; see below
    # TODO think how to do this during imputation
    for _c in _hh:
        _clean = _df.groupby(["id_household"])[_c].apply(lambda x: vote(
            x)).to_frame()
        _df = _df.drop(columns=[_c]).merge(_clean, on="id_household")

    _df = _df[_df["category_household_house_ownership"].ne(97)]
    # NOTE this category doesn't exist in Census, we have no information to compare against. This bit requires more explanation: information about household ownership unfortunately available not for all households. The 97 code is not present in the Census data meaning such records will never be selected making them useless. We retain them at the imputation stage to have more information about individuals. Two outcomes are possible and at the moment equally likely: there are other households with this code (but missing), and if we remove all those adults such households will be mislabelled as something that's not 97, another case is when there is no households with the actual label 97 (but missing), and by retaining those adults we'll be contaminating households again. This part also has to come *after* the voting, this way we make sure the whole household is removed, and not only selected people from it.
    return _df

def mutator(_df: pd.DataFrame, _map: dict = None) -> pd.DataFrame:
    for _c in _df.columns:
        _prefix = _c.split("_")[0] + "_"
        if _prefix in _map.keys():
            _df[_c] = _df[_c].astype(_map[_prefix])
    return _df
