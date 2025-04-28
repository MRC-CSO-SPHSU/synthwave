import os
import string
from pathlib import Path
from typing import Tuple

from synthwave.utils.general import path_validation, scale_sample

from synthwave.utils.uk.preprocessing import process_households
from importlib.resources import files

import pandas as pd
import numpy as np

import yaml

USOC_NAME_MAP = {}

METADATA = yaml.safe_load(files("synthwave.data.understanding_society").joinpath('name_map.yaml').read_text())

DEFAULT_OPCODES = yaml.safe_load(files("synthwave.data.understanding_society").joinpath('opcodes.yaml').read_text())

MAX_CHILDREN = yaml.safe_load(files("synthwave.data.understanding_society").joinpath('syntet.yaml').read_text())["MAX_CHILDREN"]

def format_column_name(_group: str, _native_name: str) -> str:
    def _formatter(_g: str, _dict: dict) -> str:
        if "Target" in _dict.keys():
            _prefix = f"{_g}_{_dict['Target']}_"
        else:
            _prefix = f"{_g}_person_"
        return _prefix + f"{_dict['Name']}"

    match _group:
        case "id":
            return f"id_{METADATA[_group][_native_name]['Target']}"
        case "weight":
            return f"weight_{METADATA[_group][_native_name]['Target']}"
        case "hours":
            return f"hours_person_{METADATA[_group][_native_name]['Name']}"
        case _:
            return _formatter(_group, METADATA[_group][_native_name])

for _k, _v in METADATA.items():
    for _actual_name in _v:
        USOC_NAME_MAP[_actual_name] = format_column_name(_k, _actual_name)

def convert_stata_csv(
    path_: str | Path = None, prefix: str = "a", output_dir: str | Path = None
) -> None:
    """Converts all binary Stata `.dta` files in a directory to `.csv`.

    This is a helper function to convert Understanding Society `.dta` records -
    as the most complete ones - to regular `.csv` files. The method is designed
    to work with one wave at a time.

    Parameters
    ----------
        path_ : str or Path
            The path to a directory containing `.dta` files, `None` by default.
        prefix : str
            The Understanding Society wave prefix, a single letter, `a` by
            default.
        output_dir : str | Path
            The path to a directory to store the resulting data. By default, is
            `None`, thus the data is stored in the same dir as the script
             itself.

    Notes
    -----
        Renames all columns and files by stripping them of the prefix.
    """
    p_ = path_validation(path_).expanduser()
    o_ = path_validation(output_dir).expanduser()
    assert (len(prefix) == 1) and (prefix in string.ascii_lowercase)

    for file_ in os.listdir(p_):
        if file_.startswith((f"{prefix}_", "xwavedat")):
            try:
                df_ = pd.read_stata(p_ / file_, convert_categoricals=False)
                # convert_categoricals=False do not convert data
                # to categories automatically
                df_.columns = df_.columns.str.lower().str.removeprefix(f"{prefix}_")

                if o_ is None:
                    df_.to_csv(
                        ".".join(
                            (
                                file_.removesuffix(".dta").removeprefix(f"{prefix}_"),
                                "csv",
                            )
                        ),
                        index=False,
                    )
                else:
                    df_.to_csv(
                        o_
                        / Path(
                            ".".join(
                                (
                                    file_.removesuffix(".dta").removeprefix(
                                        f"{prefix}_"
                                    ),
                                    "csv",
                                )
                            )
                        ),
                        index=False,
                    )
            except Exception as e:
                print(f"Failed to convert {file_} because of {e}")


def merge_usoc_data(
    path_: str | Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merges all provided USoc data into tables for individuals and households.

    A raw data pre-processor, this method merges multiple tables together using
    household and person ids as keys. In the case when field values differ
    individual responses take priority, unless specified otherwise.

    Parameters
    ----------
        path_ : str or Path
            The path to a directory containing .csv files, `None` by default.

    Returns
    -------
    tuple
        A tuple of two dataframes, one contains data per individual, the other
         one - per household.

    """
    p_ = path_validation(path_)

    def load_file(filename: str) -> pd.DataFrame:
        """Reads a .csv file and sets indices to 'pidp' and 'hidp'.

        Parameters
        ----------
        filename : str
            The name of a file.

        Returns
        -------
        pd.DataFrame
            A dataframe.
        """
        _fields = []
        for _, v1 in METADATA.items():
            for k2, v2 in v1.items():
                if filename in v2["File"]:
                    _fields.append(k2)

        f = pd.read_csv(
            p_ / (filename + ".csv"),
            usecols=_fields,
            engine="pyarrow",
            dtype_backend="pyarrow",
        )

        i_ = []
        if "pidp" in f.columns:
            i_.append("pidp")
        if "hidp" in f.columns:
            i_.append("hidp")

        f[i_] = f[i_].astype("uint64[pyarrow]")
        return f

    data = load_file("indall")

    def find_merge_divergent_data(old_file: pd.DataFrame, other_file: str) -> pd.DataFrame:
        """A helper function to reduce copy-paste code, does the merging.

        Combines two files into one while making sure that individual responses
        take priority over the same fields but with different values.


        Parameters
        ----------
        other_file : str
            The name of a file

        Returns
        -------
        pd.DataFrame
            The result of merging and override.

        """
        new_df = load_file(other_file)

        common_fields = set(old_file.columns) & set(new_df.columns)
        # here we identify columns for comparison

        _index = "pidp" if "pidp" in common_fields else "hidp"
        # index for merging is personal id, if not there we default to household

        common_fields -= METADATA["id"].keys()
        # sets allow us to do this without raising any exceptions

        new_df = new_df[list((set(new_df.columns) - METADATA["id"].keys()) |
                             {_index})]
        # keep non-id columns and the key

        old_file = old_file.merge(new_df, on=_index, how="left")

        def _resolver(_x, _cn):
            cx, cy = pd.isna(_x[_cn + "_x"]), pd.isna(_x[_cn + "_y"])
            if cx or cy:
                return _x[_cn + "_x"] if cy else _x[_cn + "_y"]
                # if y is na x is either na making no difference or a value
                #  else y takes precedence

            if _x[_cn + "_x"] == _x[_cn + "_y"]:
                return _x[_cn + "_x"]
            else:
                return _x[_cn + "_y"]

        for f_ in common_fields:
            old_file[f_] = old_file.apply(lambda x: _resolver(x, f_), axis=1)
            old_file = old_file.drop(columns=[f_ + "_x", f_ + "_y"])

        return old_file

    data = find_merge_divergent_data(data, "indresp")
    data = find_merge_divergent_data(data, "youth")
    data = find_merge_divergent_data(data, "child")

    _education_parents = load_file("xwavedat")
    data = (data.
            drop(columns=["maedqf", "paedqf"]).
            merge(_education_parents, how="left", on="pidp"))
    # replace education of parents with more reliable source
    #  there are other variables that we can get from xwavedat, but they are less important
    #  see https://www.understandingsociety.ac.uk/documentation/mainstage/variables/maedqf/
    income = load_file("income")[["pidp", "ficode"]]

    income = (
        income.groupby(["pidp"])["ficode"]
        .apply(set)  # multiple records from the same benefit source
        # TODO split by source?
        .apply(tuple)
        .reset_index()
    )

    data = data.merge(income, how="left", on="pidp")

    dates = (
        data.copy()
        .reset_index()[["istrtdaty", "hidp"]]
        .groupby(["hidp"])["istrtdaty"]
        .apply(tuple)
        .reset_index()
    )

    data = data.drop(columns=["istrtdaty"])

    household_response = load_file("hhresp")

    household_response = household_response.merge(
        dates, how="inner", on="hidp"
    )

    data = data.reset_index(drop=True).rename(columns=USOC_NAME_MAP)

    return data, household_response.rename(columns=USOC_NAME_MAP)

def _read_usoc_fields(
    path_: str | Path = None, verbose: bool = True
) -> dict[pd.DataFrame] | None:
    """Reads all data from provided .csv files and keeps only what's needed.

    This is a brute-force method that loops over all .csv files in a directory,
    loads them into memory one-by-one and keeps only the columns that are in the
    USOC_FIELDS list.

    Parameters
    ----------
        path_ : str or Path
            The path to a directory containing .csv files, `None` by default.
        verbose : bool
            A boolean flag that allows to print out filenames and corresponding
            columns, `True` by default.

    Notes
    -----
        This method is inherently unsafe, however, that allows it to be
        extremely flexible.

    Returns
    -------
    dict[pd.DataFrame] or None
        A dictionary with corresponding dataframes.

    """
    p_ = path_validation(path_)

    data = {}
    for filename_ in [x for x in p_.iterdir()]:
        if filename_.suffix == ".csv":

            df = pd.read_csv(
                path_ / filename_, engine="pyarrow", dtype_backend="pyarrow"
            )
            final_columns = [
                #column_ for column_ in USOC_FIELDS if column_ in df.columns
                column_ for column_ in [1] if column_ in df.columns
            ]
            data[filename_.stem] = df[final_columns]

    for k, v in data.items():
        if k != "indall":
            data[k] = v.drop(columns="gor_dv", errors="ignore")

    if verbose:
        for k, v in data.items():
            print(k)
            #print("\t", [column_ for column_ in USOC_FIELDS if column_ in v.columns])

    return data

def _find_dangling_fields(path_: str | Path = None) -> set:
    """Ensures that all requested fields are actually in USoc datasets.

    Parameters
    ----------
        path_ : str or Path
            The path to a directory containing .csv files, `None` by default.

    Returns
    -------
    set
        An empty set when everything is correct or a set of invalid column names
         otherwise.

    """
    actual_fields = []
    for k, v in _read_usoc_fields(path_, False).items():
        #actual_fields += [column_ for column_ in USOC_FIELDS if column_ in v.columns]
        actual_fields += [column_ for column_ in [1] if column_ in v.columns]
    actual_fields = set(actual_fields)
    return actual_fields.symmetric_difference([1])



def preprocess_usoc_data(_path="~/Work/data/", skip_conversion=True):
    if skip_conversion:
        individuals = pd.read_pickle(_path + "synthwave/md/individuals.pkl")
        households = pd.read_pickle(_path + "synthwave/md/households.pkl")
    else:
        convert_stata_csv(_path + "understanding_society/UKDA-6614-stata/stata/stata13_se/ukhls/",
                          output_dir=_path + "synthwave")

        individuals, households = merge_usoc_data(_path + "synthwave")

        _md_path = "synthwave/md/"
        if not os.path.exists(_path + _md_path):
            print("Preprocessing folder not found; creating at {}".format(_path + _md_path))
            os.makedirs(_path + _md_path)
        individuals.to_pickle(_path + _md_path + "individuals.pkl")
        households.to_pickle(_path + _md_path + "households.pkl")

    households = process_households(households)
    # NOTE this procedure might decrease the number of households
    individuals = individuals[individuals["id_household"].isin(households["id_household"])]
    households = households.merge(individuals.groupby(["id_household"]).size().to_frame(name="total_individuals"), on = "id_household", how="left")

    # we follow the census alternative definition: a child is anyone aged 15 and below
    children_indicator = individuals["ordinal_person_age"].lt(16)

    adults = individuals[~children_indicator].copy()
    adults = adults[sorted(adults.columns)]

    children = individuals[children_indicator].copy()[["id_household", "id_person", "indicator_person_sex", "ordinal_person_age", "category_person_ethnic_group"]]

    total_children = children.groupby(["id_household"]).agg(count=pd.NamedAgg(column='ordinal_person_age', aggfunc='count')).reset_index().rename(columns={"count": "total_children"})

    households = households.merge(total_children, how="left", on=["id_household"])
    households["total_children"] = households["total_children"].fillna(0).astype("uint8[pyarrow]")
    adults = adults.merge(households, how='left', left_on=["id_household"], right_on=["id_household"])
    # we need household attributes for the imputation stage

    non_recoverable = children[children["ordinal_person_age"].isin(DEFAULT_OPCODES.values())]["id_household"].unique()
    # the way we treat children now is based on their sex and age; we must know both to be able to build a model for prediction: 1 child age, 1 child sex, 2 child age, 2 child sex etc. We need all ages of children to actually order them
    adults = adults[~adults["id_household"].isin(non_recoverable)]
    children = children[~children["id_household"].isin(non_recoverable)]
    households = households[~households["id_household"].isin(non_recoverable)]

    for _c in adults.columns:
        if _c.startswith("category_") or _c.startswith("ordinal_"):
            adults[_c] = adults[_c].astype("int32[pyarrow]")

        if _c.startswith("hours_") or _c.startswith("income_"):
            adults[_c] = adults[_c].astype("float32[pyarrow]")

        if _c.startswith("indicator_person_vocational_qualification"):
            adults[_c] = adults[_c].astype("bool[pyarrow]")

    _e_list = ["income_person_pay", "hours_person_employment", "hours_person_overtime"]
    _se_list = ["hours_person_self_employment", "income_person_self_employment"]
    # in this survey a person is either employed, or self-employed, or neither (jobless); it can't be both employed *and* self-employed at the same time

    liars = adults[adults[_e_list].isin(DEFAULT_OPCODES.values()).all(axis=1) &
                   adults[_se_list].isin(DEFAULT_OPCODES.values()).all(axis=1) &
                   adults["income_person_second_job"].gt(0)]["id_household"]
    # TODO these people claim they have a second job, but no information about the first one is provided; we remove them for now
    adults = adults[~adults["id_household"].isin(liars)]
    children = children[~children["id_household"].isin(liars)]
    households = households[~households["id_household"].isin(liars)]

    crooked_records = adults[adults.drop(columns=["multilabel_person_benefit_income_source"]).isna().any(axis=1)]["id_person"]
    # they all are identical in terms of missing fields

    adults.loc[(~adults["id_person"].isin(crooked_records)) &
               adults['income_person_benefits'].eq(0) &
               adults['multilabel_person_benefit_income_source'].isna(), "multilabel_person_benefit_income_source"] = 0
    # zero benefits - zero source of income, keep them as a separate category

    adults.loc[(~adults["id_person"].isin(crooked_records)) &
               adults['income_person_benefits'].gt(0) &
               adults['multilabel_person_benefit_income_source'].isna(), "multilabel_person_benefit_income_source"] = 37
    # some have benefit income but no source, new category again
    # TODO read more about benefits, what if the category is missing?

    targets = adults.columns.to_list()
    targets.remove("multilabel_person_benefit_income_source")

    def split_mlb(df_):
            s = df_['multilabel_person_benefit_income_source'].explode()
            s = pd.crosstab(s.index, s).astype("bool[pyarrow]")
            s.columns = [f"mlb_{income_source}" for income_source in s.columns]
            return df_.join(s).drop(columns='multilabel_person_benefit_income_source')

    adults = split_mlb(adults)
    # being unemployed doesn't automatically entitle you to any benefits

    fact_employed = ~adults[_e_list].replace(DEFAULT_OPCODES.values(), pd.NA).isna().all(axis=1)
    fact_self_employed = ~adults[_se_list].replace(DEFAULT_OPCODES.values(), pd.NA).isna().all(axis=1)

    processed = fact_employed | fact_self_employed

    adults.loc[fact_employed, _se_list] = 0
    adults.loc[fact_employed, "indicator_person_is_employed_self_employed"] = 1
    adults.loc[fact_employed & adults["category_person_job_status"].eq(1), "category_person_job_status"] = 2

    adults.loc[fact_self_employed, _e_list] = 0
    adults.loc[fact_self_employed, "indicator_person_is_employed_self_employed"] = 2
    adults.loc[fact_self_employed & adults["category_person_job_status"].eq(2), "category_person_job_status"] = 1
    # make the data more consistent by correcting contradicting fields; we also pad with structural zeros where appropriate

    adults.loc[fact_employed, _e_list] = adults[_e_list].replace(DEFAULT_OPCODES.values(), pd.NA)
    adults.loc[fact_self_employed, _se_list] = adults[_se_list].replace(DEFAULT_OPCODES.values(), pd.NA)
    # here we know for sure that person is employed or self-employed. for such records we can completely remove OPCODEs from the corresponding part of the table.
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
    # TODO test performance

    base_mask = ~processed

    _m = base_mask & adults["indicator_person_is_employed_self_employed"].eq(1) & adults["category_person_job_status"].eq(2)
    processed |= _m
    adults.loc[_m, _se_list] = 0
    adults.loc[_m, _e_list] = adults[_e_list].replace(DEFAULT_OPCODES.values(), pd.NA)
    # stated as employed; for all people with valid combinations of indicator/job status we pad corresponding self-employment counterparts with structural zeroes

    _m = base_mask & adults["indicator_person_is_employed_self_employed"].eq(2) & adults["category_person_job_status"].eq(1)
    processed |= _m
    adults.loc[_m, _e_list] = 0
    adults.loc[_m, _se_list] = adults[_se_list].replace(DEFAULT_OPCODES.values(), pd.NA)
    # stated as self-employed; apply the same method here

    # this block should leave us with structural zeroes and NA

    jobless = (~processed & adults[_se_list + _e_list + ["indicator_person_is_employed_self_employed"]].isin([-8, -7]).all(axis=1) &
               adults[["indicator_person_did_paid_work", "indicator_person_has_job_no_work"]].isin([2, 3]).all(axis=1) &
               (~adults["category_person_job_status"].isin([1, 2])))
    adults.loc[jobless, _se_list + _e_list] = 0

    processed |= jobless
    # no indicators of any kind of job whatsoever

    shy = ~processed & adults[_se_list].isin([-1, -2]).any(axis=1) & adults["indicator_person_is_employed_self_employed"].eq(2)
    adults.loc[shy, _se_list] = adults[_se_list].replace(DEFAULT_OPCODES.values(), pd.NA)
    adults.loc[shy, _e_list] = 0
    adults.loc[shy & adults["category_person_job_status"].eq(2), "category_person_job_status"] = 1
    processed |= shy
    # where se contains -1 or -2 and indicator_person_is_employed_self_employed == 2 and category_person_job_status == 2 set category_person_job_status 1

    shy = ~processed & adults[_e_list].isin([-1, -2]).any(axis=1) & adults["indicator_person_is_employed_self_employed"].eq(1)
    adults.loc[shy, _e_list] = adults[_e_list].replace(DEFAULT_OPCODES.values(), pd.NA)
    adults.loc[shy, _se_list] = 0
    adults.loc[shy & adults["category_person_job_status"].eq(1), "category_person_job_status"] = 2
    processed |= shy
    # at this stage all options stemming from _e_list and _se_list have been exhausted

    # this stems from the data analysis. codes -1 and -2 usually mean proxy or something else like this, where people don't want or can't answer this question clearly. this, however, serves as an indicator they are self-/employed. we simply make the records more consistent; they are self-contradicting otherwise

    adults = adults.replace([-9, -8, -7], pd.NA)
    processed |= adults[["indicator_person_is_employed_self_employed", "indicator_person_did_paid_work", "indicator_person_has_job_no_work", "category_person_job_status"]].isna().all(axis=1)
    # just records with no information about them

    leftovers = (adults[["indicator_person_is_employed_self_employed", "indicator_person_did_paid_work", "indicator_person_has_job_no_work"]].isna().all(axis=1) &
                 adults["category_person_job_status"].eq(1))
    processed |= leftovers
    adults.loc[leftovers, _e_list] = 0
    adults.loc[leftovers, "indicator_person_is_employed_self_employed"] = 2

    leftovers = (adults[["indicator_person_is_employed_self_employed", "indicator_person_did_paid_work", "indicator_person_has_job_no_work"]].isna().all(axis=1) &
                 adults["category_person_job_status"].eq(2))
    processed |= leftovers
    adults.loc[leftovers, _se_list] = 0
    adults.loc[leftovers, "indicator_person_is_employed_self_employed"] = 1
    # here we rely only on category_person_job_status

    _mask = ~processed & adults["category_person_job_status"].gt(2)
    adults.loc[_mask, "indicator_person_is_employed_self_employed"] = pd.NA
    # this leaves us with little to no information about employment/self-employment in particular; we just remove any non-na fields
    processed |= _mask

    _mask = ~processed & adults["indicator_person_is_employed_self_employed"].isna() & adults["category_person_job_status"].eq(-2)
    adults.loc[_mask, "category_person_job_status"] = pd.NA
    processed |= _mask

    leftovers2 = ~processed & adults["category_person_job_status"].isin([1, 2]) & adults[["indicator_person_did_paid_work", "indicator_person_has_job_no_work"]].isin([2, 3]).all(axis=1)
    adults.loc[leftovers2, ["indicator_person_is_employed_self_employed", "category_person_job_status"]] = pd.NA
    processed |= leftovers2
    # when work indicators are 2, 3 we can't rely on job status so we pad it

    # now category_person_job_status is 1 and 2 only, we just fill in indicator_person_is_employed_self_employed
    adults.loc[~processed & adults["category_person_job_status"].eq(1), "indicator_person_is_employed_self_employed"] = 2
    adults.loc[~processed & adults["category_person_job_status"].eq(1), _e_list] = 0

    adults.loc[~processed & adults["category_person_job_status"].eq(2), "indicator_person_is_employed_self_employed"] = 1
    adults.loc[~processed & adults["category_person_job_status"].eq(2), _se_list] = 0

    adults = adults.replace(DEFAULT_OPCODES.values(), pd.NA)
    adults["indicator_person_sex"] = adults["indicator_person_sex"].map({1: True, 2: False}).astype("bool[pyarrow]")

    adults = adults.drop(columns=["indicator_person_did_paid_work", "indicator_person_has_job_no_work"])

    def is_employed(_x):
        if pd.isna(_x):
            return _x
        return True if _x == 1 else False

    def is_self_employed(_x):
        if pd.isna(_x):
            return _x
        return True if _x == 2 else False

    adults["indicator_person_is_employed"] = adults["indicator_person_is_employed_self_employed"].map(is_employed).astype("bool[pyarrow]")
    adults["indicator_person_is_self_employed"] = adults["indicator_person_is_employed_self_employed"].map(is_self_employed).astype("bool[pyarrow]")

    # TODO investigate all and skipna
    _valid = ~adults[_se_list + _e_list].isna().any(axis=1)
    adults.loc[_valid & adults[_se_list + _e_list].eq(0).all(axis=1), ["indicator_person_is_employed", "indicator_person_is_self_employed"]] = False

    adults = adults.drop(columns=["indicator_person_is_employed_self_employed", "mlb_0"])

    adults.loc[((adults["indicator_person_is_employed"].isna() & adults["indicator_person_is_self_employed"].isna()) | (adults["indicator_person_is_employed"] | adults["indicator_person_is_self_employed"])) & adults["category_person_job_sic"].eq(0), "category_person_job_sic"] = pd.NA
    # these are spurious records, code zero is sensible for people without a job only

    adults.loc[adults["category_person_job_sic"].eq(0), "category_person_job_nssec"] = 0

    assert (adults.nunique() > 1).all()
    # make sure there is no degenerate distributions; when there is only one value there is no need to do imputation, you just pad empty fields with the same value

    adults = adults.replace([None, np.nan], pd.NA) # do only once at the end
    # TODO explore behaviour of this bit: when calculating rates this bit doesn't replace nans with pd.NA

    adults["category_household_type"] = pd.NA

    valid_singles = adults[(adults["total_individuals"] - adults["total_children"]).eq(1)]["id_household"]
    adults.loc[adults["id_household"].isin(valid_singles), "category_household_type"] = adults[adults["id_household"].isin(valid_singles)]["total_children"].map(lambda x: f"a{x}")
    assert adults[adults["id_household"].isin(valid_singles)]["id_partner"].isna().all()

    one_couple = adults[adults["id_person"].isin(adults["id_partner"])].groupby("id_household").size().to_frame(name="people_in_couples").reset_index()
    one_couple = one_couple[one_couple["people_in_couples"].eq(2)]["id_household"].unique()

    valid_couples = adults["id_household"].isin(one_couple) & (adults["total_individuals"] - adults["total_children"]).eq(2)
    adults.loc[valid_couples, "category_household_type"] = adults[valid_couples]["total_children"].map(lambda x: f"c{x}")
    assert not adults[valid_couples]["id_partner"].isna().any()

    multiple_adults = adults[adults["category_household_type"].isna() & adults["total_children"].eq(0)]["id_household"].unique()
    adults.loc[adults["id_household"].isin(multiple_adults), "category_household_type"] = adults[adults["id_household"].isin(multiple_adults)]["total_individuals"].map(lambda x: f"m{x}")

    adults.loc[adults["category_household_type"].isna(), "category_household_type"] = "mf"

    adults["has_partner"] = False
    adults.loc[adults["id_person"].isin(adults["id_partner"]), "has_partner"] = True

    adults["minutes_person_employment"] = -1
    adults["minutes_person_employment"] = adults["minutes_person_employment"].astype("float[pyarrow]")

    _mask = ~adults["hours_person_employment"].isna()
    adults.loc[_mask, "fp"] = adults[_mask]["hours_person_employment"].mul(100).round().astype(float).div(100).mod(1).mul(100).astype(int)
    adults["fp"] = adults["fp"].fillna(0)

    adults.loc[adults["fp"].eq(0) | adults["fp"].eq(50) | adults["fp"].eq(60) | adults["fp"].gt(65), "minutes_person_employment"] = adults["hours_person_employment"] * 60
    adults.loc[adults["fp"].eq(45) | adults["fp"].eq(44), "minutes_person_employment"] = adults[adults["fp"].eq(45) | adults["fp"].eq(44)]["hours_person_employment"].astype(int) * 60 + 45 # this one supposed to round down; dirty trick to avoid dealing with empty values
    adults.loc[adults["fp"].eq(15) | adults["fp"].eq(14) | adults["fp"].eq(25) | adults["fp"].eq(23), "minutes_person_employment"] = adults[adults["fp"].eq(15) | adults["fp"].eq(14) | adults["fp"].eq(25) | adults["fp"].eq(23)]["hours_person_employment"].astype(int) * 60 + 15 # see comment above
    adults.loc[adults["fp"].eq(5) , "minutes_person_employment"] = adults[adults["fp"].eq(5)]["hours_person_employment"].astype(int) * 60 + 5
    adults.loc[adults["fp"].eq(32) | adults["fp"].eq(19) | adults["fp"].eq(20) , "minutes_person_employment"] = adults[adults["fp"].eq(32) | adults["fp"].eq(19) | adults["fp"].eq(20)]["hours_person_employment"].astype(int) * 60 + 20
    adults.loc[adults["fp"].eq(30) | adults["fp"].eq(29), "minutes_person_employment"] = adults[adults["fp"].eq(30) | adults["fp"].eq(29)]["hours_person_employment"].astype(int) * 60 + 30
    adults.loc[adults["fp"].eq(39) | adults["fp"].eq(65), "minutes_person_employment"] = adults[adults["fp"].eq(39)]["hours_person_employment"].astype(int) * 60 + 40

    adults = adults.drop(columns=["fp"]).astype({"minutes_person_employment": "int16[pyarrow]"})

    # NOTE in wave 1 hours_person_overtime & hours_person_self_employment is all round hours, no need to do anything

    rate = (adults["income_person_pay"] / (13 * (adults["hours_person_employment"] + adults["hours_person_overtime"]) / 4)).astype(float).astype("float[pyarrow]")

    adults.loc[rate < 2.5, "income_person_pay"] = pd.NA
    adults.loc[rate < 2.5, "minutes_person_employment"] = pd.NA
    adults = adults.drop(columns=["hours_person_employment"])
    # a fix for records that show absurdly low hourly rate
    # TODO is there a better way to do this? this rate depends on the year of the survey, age and probably some other aspects; overtime adds even more complexity

    incomplete_qualification = adults[[f"indicator_person_vocational_qualification_{i}" for i in ([j for j in range(1, 16)] + [96])]].isna().any(axis=1)
    # at this stage we deal with complete vocational qualification records separately
    # such records are full of NA only

    # for complete records only, the semantics of vq_96 "None of the above" means other qualifications must be false.
    # there is a very limited number of records in this case when vq_96 == True, matching three patterns only:
    #  - vq_96 == True and all other vq_ are False, no need to do anything
    #  - *all* qualifications are true which is highly unlikely; we pad everything that's not 96 with False
    #  - some of them are true, we give those records the benefit of the doubt and pad only 96 with False
    adults.loc[~incomplete_qualification &
               adults[[f"indicator_person_vocational_qualification_{i}" for i in [j for j in range(1, 16)] + [96]]].all(axis=1),
    [f"indicator_person_vocational_qualification_{i}" for i in [j for j in range(1, 16)]]] = False

    adults.loc[~incomplete_qualification &
                   adults[[f"indicator_person_vocational_qualification_{i}" for i in [j for j in range(1, 16)]]].any(axis=1) &
                   adults["indicator_person_vocational_qualification_96"], "indicator_person_vocational_qualification_96"] = False

    # At this stage vq 96 doesn't play any role (it's False when any vq is true, and True when all of them are false)
    adults = adults.drop(columns=["indicator_person_vocational_qualification_96"], errors="ignore")

    adults.loc[adults["hours_person_self_employment"].eq(0) &
               adults["income_person_self_employment"].eq(0) &

               adults["minutes_person_employment"].eq(0) &
               adults["income_person_pay"].eq(0) &
               adults["hours_person_overtime"].eq(0) &

               adults["category_person_job_nssec"].eq(0) &
               adults["category_person_job_sic"].eq(0) &

               ~adults["category_person_job_status"].isin([1, 2]) &

               ~adults["indicator_person_is_employed"] &
               ~adults["indicator_person_is_self_employed"], "income_person_second_job"] = 0
    # Padding with structural zeroes to make sure people with no main job don't have a second one

    adults["indicator_household_has_central_heating"] = adults["indicator_household_has_central_heating"].map({1: True, 2: False}).astype("bool[pyarrow]")

    adults = scale_sample(adults, "weight_household")

    _p = path_validation(_path).expanduser()

    adults.to_parquet(_p / "adults_non_imputed_middle_fidelity.parquet",
                        index=False,
                        )

    children = children.replace(DEFAULT_OPCODES.values(), pd.NA)
    children["indicator_person_sex"] = children["indicator_person_sex"].map({1: True, 2: False}).astype("bool[pyarrow]")
    children.to_parquet(_p / "children_non_imputed_middle_fidelity.parquet",
                        index=False,
                        )

    households = households.replace(DEFAULT_OPCODES.values(), pd.NA)
    households.to_parquet(_p / "households_non_imputed_middle_fidelity.parquet",
                        index=False,
                        )
