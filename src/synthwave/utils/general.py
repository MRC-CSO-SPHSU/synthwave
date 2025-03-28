from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def path_validation(path_: Path | str) -> Path:
    """Ensures that the argument is a valid Path object or string.

    Runs a series of checks to make sure there is no problems with a file path.

    Parameters
    ----------
        path_ : str, Path
            The path to a directory.

    Returns
    -------
    Path
        The input Path, or the input string converted to one.

    Raises
    ------
    TypeError
        When the input is `None`, or not a string/Path object.

    """
    if path_ is None:
        raise TypeError('Provided path is None.')

    if not (isinstance(path_, str) or isinstance(path_, Path)):
        raise TypeError(f'Provided path is not a string/Path object but'
                        f' {type(path_)}.')

    return path_ if isinstance(path_, Path) else Path(path_)


def get_formatted_attributes(ids_: str = 'id_household',
                             person_misc: tuple[str, ...] | None = None,
                             person_category: tuple[str, ...] | None = None,
                             household_misc: tuple[str, ...] | None = None,
                             household_category: tuple[str, ...] | None = None) -> dict:
    if not isinstance(ids_, str):
        raise TypeError("Not a correct parameter")

    s_ = [ids_]

    for l_ in [person_misc, person_category, household_misc, household_category]:
        if not isinstance(l_, tuple):
            raise TypeError("Not a correct parameter")
        for v_ in l_:
            if not isinstance(v_, str):
                raise TypeError("Not a correct parameter")

        s_ = s_ + list(l_)

    assert len(set(s_)) == len(s_)

    attributes = dict(person=dict(), household=dict())

    attributes["person"]["misc"] = list(person_misc)
    attributes["person"]["category"] = list(person_category)
    attributes["household"]["misc"] = list(household_misc)
    attributes["household"]["category"] = list(household_category)
    attributes["household"]["id"] = ids_

    return attributes


def remove_degenerate_columns(target_df: pd.DataFrame,
                              predictor_list: list[str, ...]) -> list[str, ...]:
    dc = target_df.apply(lambda x: len(pd.unique(x))).to_frame("n_unique")

    for c_ in dc[dc["n_unique"] == 1].index.to_list():
        if c_ in predictor_list:
            predictor_list.remove(c_)

    return predictor_list

def  remove_identical_columns(target_df: pd.DataFrame):
    truth_table = (pd.concat([target_df.apply(lambda x: x.equals(target_df[name_])) for name_ in target_df.columns], axis=1)
                   .rename(columns={index_: name_ for index_, name_ in enumerate(target_df.columns)}))

    for name_ in target_df.columns:
        truth_table.loc[name_: , name_] = False
    # remove self-comparisons and symmetry-caused copies
    target_df = target_df.loc[:, (~truth_table.apply(lambda x: x.any(), axis=0).values).tolist()]
    return target_df
    # TODO check this version of the code, there were several versions

def mutator(_df: pd.DataFrame) -> pd.DataFrame:
    _columns = _df.columns
    _map = {"ordinal_": "uint8[pyarrow]",
            "total_": "uint8[pyarrow]",
            "category_": "uint8[pyarrow]",
            "hours_": "float32[pyarrow]",
            "income_": "float32[pyarrow]",
            "indicator_": "bool[pyarrow]",
            "mlb_": "bool[pyarrow]",
            }
    for _c in _columns:
        _prefix = _c.split("_")[0] + "_"
        if _prefix in _map.keys():
            _df[_c] = _df[_c].astype(_map[_prefix])
    return _df

def cast_type():
    # if category has only two values that's an indicator
    pass

def convert_types():
    pass

def is_clean(_column: pd.Series, _codes: list) -> bool:
    return False if _column.isin(_codes).any() else True

def generate_household_id(_sample_size: int = 10_000_000,
                          _mini_batch_id: int = None,
                          _micro_batch_id: int = None,
                          _region_id: int = None,
                          _household_type: int = None) -> pd.Series:
    """

    The household id has the following structure:
    xx xx x xx xxxxxxx xx
    |  |  | |  |      |
    |  |  | |  |      two buffer digits to allow for personal ids based on the household id itself, up to double digits (0 - 99); zeroes by default
    |  |  | |  seven digits (0 - 9 999 999) to differentiate between records within the same sample; the sample can't be more than 10 000 000 households in total
    |  |  | micro-batch id limited to two digits (0 - 99)
    |  |  mini-batch id limited to one digit (0 - 9)
    |  two digits for the region id (1 - 99)
    household type id (0 - 99), two digits again


    Parameters
    ----------
    _sample_size
    _mini_batch_id
    _micro_batch_id
    _region_id
    _household_type

    Returns
    -------

    """
    assert 1 <= _sample_size <= 10_000_000_000
    assert 0 <= _micro_batch_id <= 99
    assert 0 <= _mini_batch_id <= 9
    assert 1 <= _region_id <= 99
    assert 0 <= _household_type <= 99

    def _build_id_base():
        return (_household_type * 1_00_0_00_0000000_00 +
                        _region_id * 1_0_00_0000000_00 +
                      _mini_batch_id * 1_00_0000000_00 +
                        _micro_batch_id * 1_0000000_00)

    return pd.Series(
        range(_build_id_base(), _build_id_base() + _sample_size * 100, 100))

def generate_personal_ids(_df: pd.DataFrame, contains_couples=False) -> pd.DataFrame:
    # at this stage household ids have been re-integrated into the dataset, and it is in the long format
    # by data design if there is a couple in a household it is always two first individuals
    # this means their personal ids end with 0 and 1
    # and corresponding partners ids end with 1 and 0, respectively
    if contains_couples:
        return (_df.assign(id_cumulative = lambda x: x[["id_household"]].groupby(["id_household"]).cumcount(),
                           id_person = lambda x: x["id_household"] + x["id_cumulative"],
                           id_partner = lambda x: x["id_household"] + 1 - x["id_cumulative"].where(x["id_cumulative"] <= 1, pd.NA)).
                drop(columns=["id_cumulative"]).
                astype({'id_partner': 'uint64[pyarrow]'}))
    else:
        return _df.assign(id_person = lambda x: x["id_household"] + x[["id_household"]].groupby(["id_household"]).cumcount(),
                          id_partner = pd.NA)

def build_correlation_heatmap(df,
                              default_prefixes: tuple[str, ...] = ("ordinal_", "income_", "total_", "category_", "indicator_"),
                              no_transform: tuple[str, ...] = ("indicator_", "category"),
                              to_sort=True):
    _df_local = df[[_c for _c in df.columns if _c.startswith(default_prefixes)]].copy()

    for c in _df_local.columns:
        if c.startswith(default_prefixes) and not c.startswith(no_transform):
            _df_local[c] = _df_local[c].astype("int32[pyarrow]")
            _df_local[c + "_squared"] = _df_local[c] * _df_local[c]
            _df_local[c + "_cubed"] = _df_local[c + "_squared"] * _df_local[c]

    plt.figure(figsize=(40, 40))
    correlation = _df_local.corr()

    if to_sort:
        sum_corr = correlation.sum().sort_values(ascending=True).index.values

        return sns.heatmap(_df_local[sum_corr].corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG', fmt=".1g", square=True)
    else:
        return sns.heatmap(correlation, vmin=-1, vmax=1, annot=True, cmap='BrBG', fmt=".1g", square=True)

def scale_sample(df: pd.DataFrame, scale_column: str | None = None) -> pd.DataFrame | None:
    if scale_column is None:
        return None
    else:
        return (df
                .reindex(df
                         .index
                         .repeat(df[scale_column])
                         )
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
                .drop(columns=scale_column)
                )
