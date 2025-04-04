from typing import Tuple, Union, Iterable

import pandas as pd

from collections import Counter

import yaml

from importlib.resources import files
DEFAULT_OPCODES = yaml.safe_load(files("synthwave.data.understanding_society").joinpath('opcodes.yaml').read_text())

import logging
logger = logging.getLogger(__name__)

def filter_dates(raw_dates: Tuple) -> Union[float, pd.NA]:
    # intentionally disregard OPCODES
    # TODO get rid of the union
    _clear_dates = []
    for _date in raw_dates:
        if not pd.isna(_date) and not (_date in DEFAULT_OPCODES.values()):
            _clear_dates.append(_date)

    match len(_clear_dates):
        case 0: return pd.NA
        case 1: return _clear_dates[0]
        case _: return Counter(_clear_dates).most_common(1)[0][0]

def replace_missing_values(_df: pd.DataFrame,
                        _map: None | dict[str, tuple[int, int] | Iterable[tuple[int, int]]]) -> pd.DataFrame:
    # NOTE order is important
    if _map is None:
        return _df.replace(DEFAULT_OPCODES.values(), pd.NA)

    for k, v in _map.items():
        if isinstance(v, tuple):
            _df[k] = _df[k].replace(v[0], v[1])
        else:
            for m in v:
                _df[k] = _df[k].replace(m[0], m[1])

    return _df.replace(DEFAULT_OPCODES.values(), pd.NA)

def process_households(_df: pd.DataFrame) -> pd.DataFrame:
    degenerate_households = _df["weight_household"] <= 0
    # we don't care about the reason, we discard them anyway
    _log_msg = (f'Discarding n={degenerate_households.sum()} '
                f'households with invalid weights')
    logger.info(_log_msg)

    _df = _df[_df["weight_household"] > 0]

    factor_ = max(_df["weight_household"]) / (3 * min(_df["weight_household"]))
    # FIXME this factor of 3 is introduced to reduce the dataset size.
    #  R can't handle this much due to integer overflow

    _df = (_df.
           assign(weight_household=lambda x: round(x.weight_household * factor_)).
           astype({'weight_household': 'uint16[pyarrow]'}))

    _df["ordinal_household_year"] = _df["ordinal_household_year"].map(filter_dates) - 2000

    _missing_dates = _df["ordinal_household_year"].isna().sum() * 100 / len(_df.index)

    logger.info(f"{_missing_dates} households miss the interview date")
    return _df
