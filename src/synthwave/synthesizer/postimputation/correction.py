from tqdm import tqdm

from synthwave.synthesizer.uk.constraints import validate
from synthwave.synthesizer.utils import clean_imputed_data
import pandas as pd

def get_patterns(_df, column_list):
    _mask = _df[column_list].isna().drop_duplicates()
    _mask = _mask[_mask.any(axis=1)]
    return _mask.reset_index(drop=True)


def correct_imputed_data(_df):

    # FIXME this is a temporary workaround here to be able to employ constraints, it should be done elsewhere.
    _adjusted_time = [_c for _c in _df.columns if "minutes" in _c]
    _df[_adjusted_time] = _df[_adjusted_time].div(15).round(0).mul(15).astype("uint16[pyarrow]")

    _df = clean_imputed_data(_df)

    # TODO we leave job classification to simulated annealing and synthesis; the former does nation-wide adjustment, the latter restricts generation only to combinations that are already there.

    # TODO imputation affects hourly rates, but we don't check them atm

    _df["indicator_person_requires_correction"] = ~validate((
        "indicator_person_is_self_employed",
        "indicator_person_is_employed",

        "minutes_person_employment",
        "income_person_pay",
        "hours_person_overtime",

        "hours_person_self_employment",
        "income_person_self_employment",

        "category_person_job_nssec",
        "category_person_job_sic",

        "category_person_job_status",

        "income_person_second_job"), _df)

    _trouble_counts = _df[_df["indicator_person_requires_correction"]][["id_person"]].value_counts().reset_index()

    _people_counts = (_df[_df["id_person"].isin(_df[_df["indicator_person_requires_correction"]]["id_person"].unique())][["id_person"]].
                      value_counts().
                      reset_index().
                      rename(columns={"count": "total_individuals"}).
                      merge(_trouble_counts, on="id_person", how="inner").
                      rename(columns={"count": "corrupted_records"}).
                      assign(sampling_pool=lambda x: x.total_individuals - x.corrupted_records).
                      assign(pool_ratio=lambda x: x.sampling_pool / x.corrupted_records))

    to_transplant = _people_counts[_people_counts["pool_ratio"].ge(0.1)]["id_person"]

    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import NearestNeighbors

    num_pipe = make_pipeline(MinMaxScaler())
    cat_pipe = make_pipeline(OneHotEncoder(handle_unknown="error", sparse_output=False))

    _df["id_person_new"] = range(1, len(_df) + 1)

    preprocessor_tree = make_column_transformer(
        (num_pipe, ["minutes_person_employment", "income_person_pay", "hours_person_overtime",
                    "hours_person_self_employment", "income_person_self_employment", "income_person_second_job"]),
        (cat_pipe, ["category_person_job_nssec", "category_person_job_sic", "category_person_job_status"]),
        remainder='passthrough',
        n_jobs=8,
        force_int_remainder_cols=False
    )

    _target_columns = ["indicator_person_is_self_employed", "indicator_person_is_employed",
                       "minutes_person_employment", "income_person_pay", "hours_person_overtime",
                       "hours_person_self_employment", "income_person_self_employment",
                       "category_person_job_nssec", "category_person_job_sic",
                       "category_person_job_status", "income_person_second_job"]
    # run across *all* records
    preprocessor_tree.fit(_df[_target_columns])

    # parallel
    # from multiprocessing import Pool
    # pool = Pool(8)
    #
    # # split into list of frames grouped by id
    # #
    #
    # # move all invalid records and donors based on the same id to another frame.
    # correction_df = adults[adults["id_person"].isin(to_transplant)].copy()
    #
    # # keep valid records in the old dataframe
    # ttt_adults = adults[~adults["id_person"].isin(to_transplant)].copy()
    #
    # clones = [g for _, g in correction_df.groupby('id_person')]
    # # NOTE can't believe there is no aesthetically more pleasant approach
    #
    # def surgery(_clones):
    #     _recipients = _clones[_clones["indicator_person_requires_correction"]][_target_columns + ["id_person_new"]]
    #     _donors = _clones[~_clones["indicator_person_requires_correction"]][_target_columns]
    #
    #     knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    #
    #     _ready_to_donor = preprocessor_tree.transform(_donors)
    #
    #     knn.fit(_ready_to_donor)
    #
    #     _final_donors = knn.kneighbors(preprocessor_tree.transform(_recipients.drop(columns=["id_person_new"])), return_distance=False)
    #
    #     _mask = _clones["id_person_new"].isin(_recipients["id_person_new"])
    #
    #     _transplants = _donors.iloc[_final_donors.flatten(), :].reset_index(drop=True)
    #     # TODO make a test to ensure this works as intended
    #
    #     _clones.loc[_mask, _target_columns] = _transplants[_target_columns].values
    #     # TODO see https://stackoverflow.com/questions/39267372/replace-rows-in-a-pandas-df-with-rows-from-another-df for inconsistent behaviour
    #     _clones.loc[_mask, "indicator_person_requires_correction"] = False
    #     return _clones
    #
    # a = [pool.apply_async(surgery, kwds={"_clones": clone_}) for clone_ in clones]
    #
    # for i in a:
    #     i.wait()

    for _person in tqdm(to_transplant):
        _recipients = _df[_df["id_person"].eq(_person) &
                             _df["indicator_person_requires_correction"]][_target_columns + ["id_person_new"]]

        _donors = _df[_df["id_person"].eq(_person) &
                         ~_df["indicator_person_requires_correction"]][_target_columns]

        knn = NearestNeighbors(n_neighbors=1,
                               algorithm='ball_tree',
                               n_jobs=8)

        _ready_to_donor = preprocessor_tree.transform(_donors)

        knn.fit(_ready_to_donor)

        _final_donors = knn.kneighbors(preprocessor_tree.transform(_recipients.drop(columns=["id_person_new"])), return_distance=False)

        _mask = _df["id_person_new"].isin(_recipients["id_person_new"])

        _transplants = _donors.iloc[_final_donors.flatten(), :].reset_index(drop=True)
        # TODO make a test to ensure this works as intended

        _df.loc[_mask, _target_columns] = _transplants[_target_columns].values
        # TODO see https://stackoverflow.com/questions/39267372/replace-rows-in-a-pandas-df-with-rows-from-another-df for inconsistent behaviour
        _df.loc[_mask, "indicator_person_requires_correction"] = False
    # NOTE this is a pretty crude version of what should be; it leads to excessive repetitions in the block of given variables. more variations/iterations of imputations should alleviate this

    for c_ in _df.columns:
        # this approach keeps zeroes and doesn't affect small incomes at this stage any more
        if "income" in c_:
            positive_income = _df[c_].le(10) & _df[c_].gt(0)
            _df.loc[positive_income, c_] = 10

            negative_income = _df[c_].ge(-10) & _df[c_].lt(0)
            _df.loc[negative_income, c_] = -10

            _df[c_] = _df[c_].round(-1)
            # TODO investigate how early coarse-graining of income affects the transplant stage

    # FIXME removing records that haven't been processed properly is a temporary workaround
    _unfixable_individuals = _df[_df["indicator_person_requires_correction"]]["id_person"]
    _unfixable_households = _df[_df["id_person"].isin(_unfixable_individuals)]["id_household"]
    _df = _df[~_df["id_household"].isin(_unfixable_households)]

    _df = _df.drop(columns=["indicator_person_requires_correction", "id_person_new"])
    # NOTE even though the imputation stage might introduce some (improbable) combinations of parameters that do not exist in reality we disregard them. The process of annealing will alleviate this.

    _df.loc[_df["category_household_type"].isin(['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a9',]), "category_household_type"] = "a1+"
    _df.loc[_df["category_household_type"].isin(['c3', 'c4', 'c5', 'c6', 'c7', 'c8']), "category_household_type"] = "c3+"

    # m2 is 1 adult and 1 pseudo-adult, no couples
    # m3 and m4 can have one couple inside, their records must go first, make sure the ages are ordered and pass the check

    def mark_households_with_couples(_df: pd.DataFrame, _hh_type: str) -> pd.Series:

        if _hh_type not in (_types := ["m3", "m4"]):
            raise ValueError(f"Household type must be either {_types[0]} or {_types[1]}")

        _no_couples_mask = _df[_df["category_household_type"].eq(_hh_type)][["id_household", "id_person", "id_partner"]].drop_duplicates().groupby(["id_household"])["id_partner"].apply(lambda x: x.isna().all()).reset_index()
        # this way we mark m3 households with no couples inside i.e., one adult and two pseudo-adults
        _couples = _no_couples_mask[~_no_couples_mask["id_partner"]]["id_household"]
        return _df["id_household"].isin(_couples)


    _df.loc[mark_households_with_couples(_df, "m3"), "category_household_type"] = "mc3"
    _df.loc[mark_households_with_couples(_df, "m4"), "category_household_type"] = "mc4"

    # mf is mix and match
    # total_individuals == 3, total_children == 1 # that is 1 adult and 1 pseudo-adult, otherwise that would be a couple
    # total_individuals == 4, total_children == 1 # can be a couple inside + 1 pseudo-adult; or 1 adult and two pseudo-adults; needs a split
    # TODO do the same for mf

    #_test = adults[adults["category_household_type"].isin(["mf"]) & adults["total_adults"].eq(2) & adults["total_children"].eq(1)].copy()
    return _df
