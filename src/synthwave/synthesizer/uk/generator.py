import os.path

from sklearn.dummy import DummyClassifier
import logging

from synthwave.utils.general import generate_household_id, generate_personal_ids
from synthwave.utils.procrustes import Procrustes

logger = logging.getLogger(__name__)

import pyarrow as pa
from rdt.transformers import FloatFormatter
from sdv.single_table import CTGANSynthesizer
from synthwave.synthesizer.utils import metadata_constructor

from synthwave.synthesizer.uk.constraints import *
from sdv.constraints import create_custom_constraint_class

import re
import yaml

import joblib

from synthwave.synthesizer.axolotl_tank import get_optimal_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from importlib.resources import files

HOUSEHOLD_ID_MAP = {_v: _i  for _i, _v in enumerate(["a0", "a1+", "c0", "c1", "c2", "c3+", "m2", "m3", "mc3", "m4", "mc4"])}
# their relative order is irrelevant at the moment as long as it is the same across all data

MAX_CHILDREN = yaml.safe_load(files("synthwave.data.understanding_society").joinpath('syntet.yaml').read_text())["MAX_CHILDREN"]


class Syntets:
    DEFAULT_GROUPS = ["a0", "a1+", "c0", "c1", "c2", "c3+", "m2", "m3", "mc3", "m4", "mc4"]
    def __init__(self, raw_data):
        self.raw_data = raw_data.copy()
        self.data = self.raw_data.copy()
        self.household_types = raw_data["category_household_type"].unique()

        self.groups = dict()
        self.subsets = []

    @staticmethod
    def generate_schema(_df: pd.DataFrame):
        _s = []

        for _c in _df.columns:
            if _c.startswith(("indicator_", "mlb_")):
                _s.append((_c, pa.bool_()))
            if _c.startswith(("ordinal_", "category_", "hours_")):
                _s.append((_c, pa.uint8()))
            if _c.startswith("minutes_"):
                _s.append((_c, pa.uint16()))
            if _c.startswith("income_"):
                if _c == "income_person_self_employment":
                    _s.append((_c, pa.int16()))
                elif _c == "income_person_investment":
                    _s.append((_c, pa.uint32()))
                else:
                    _s.append((_c, pa.uint16()))
        return pa.schema(_s)

    def convert_types(self):
        # unfortunately, ctgan still struggles to deal well with modern types, we replace all of them
        for _g in self.subsets:
            _columns = self.groups[_g]["data"].columns
            _int_target = []

            for _c in _columns:
                if _c.split("_")[0] in ["category", "hours", "ordinal", "income", "ordinal", "total", "minutes"] and _c != "category_household_type":
                    _int_target.append(_c)

            self.groups[_g]["data"][_int_target] = self.groups[_g]["data"][_int_target].astype(int)

            _bools = [_c for _c in _columns if _c.startswith(("indicator_", "mlb_"))]
            self.groups[_g]["data"][_bools] = self.groups[_g]["data"][_bools].astype(bool)

    def split_data(self):
        for _g in Syntets.DEFAULT_GROUPS:
            d = self._splitter(self.data, [_g])
            if not _g.endswith("+"):
                d = d.drop(columns=["total_children"])
            if len(d) > 0:
                self.subsets.append(_g)
                self.groups[_g] = {"data": d, "model": None, "dropouts": None}

    @staticmethod
    def _splitter(_df: pd.DataFrame, _type: list) -> pd.DataFrame:
        # select a subset
        return (_df[_df["category_household_type"].isin(_type)]
                .copy()
                .drop(columns=["category_household_type",
                               ]).reset_index(drop=True))
    # we don't need household type any more, another hh classification will be introduced at the post-synthesis stage

    def locate_degenerate_distributions(self):
         # degeneracy might appear when we convert from long to wide
        for _g in self.subsets:
            if self.groups[_g]["data"] is None:
                logger.warning(f"Dataset {_g} is missing")
                continue
            else:
                _combinations = self.groups[_g]["data"].apply(pd.unique)
                _exclude = _combinations[_combinations.map(len) == 1].map(lambda x: x[0]).to_dict()
                self.groups[_g]["dropouts"] = _exclude
                self.groups[_g]["data"].drop(columns=_exclude.keys(), inplace=True)

                _cmb = _combinations[_combinations.index.str.startswith("category_") &
                                     _combinations.map(len).eq(2)]
                if len(_cmb) > 0:
                    _msg = f"In household {_g} columns {_cmb.index} are potential candidates for remapping"
                    logger.warning(_msg)

    @staticmethod
    def _get_household_attributes(_df: pd.DataFrame) -> list[str, ...]:
        _columns = []
        for _c in _df.columns:
            if "household" in _c and not _c.startswith("id_"):
                _columns.append(_c)
        return _columns

    def restructure_data(self):
        for _g in self.subsets:
            _columns_household = self._get_household_attributes(self.groups[_g]["data"])

            if _g.endswith("+"):
                _columns_household.append("total_children")

            household_attributes = self.groups[_g]["data"][_columns_household + ["id_household"]].copy().drop_duplicates()

            self.groups[_g]["data"].drop(columns=_columns_household, inplace=True)

            if not _g.startswith("a"):
                if _g.startswith("c"):
                    self.groups[_g]["data"].sort_values(by=["id_household",
                                                            'ordinal_person_age', # now first person is always older, but sex can vary
                                                            'indicator_person_sex'], inplace=True, ascending=False)
                elif _g.startswith("m"):
                    if "c" in _g:
                        self.groups[_g]["data"]["is_in_couple"] = self.groups[_g]["data"]["id_partner"].notna()
                        self.groups[_g]["data"] = self.groups[_g]["data"].drop(columns="id_partner")
                        self.groups[_g]["data"] = Procrustes.stretch_wide_adults(self.groups[_g]["data"].
                                                                                       sort_values(["id_household", "is_in_couple", "ordinal_person_age"], ascending=False).
                                                                                       drop(columns="is_in_couple"))
                        # for mc3/mc4 the rules are as follows:
                        #  - the eldest person in couple comes first
                        #  - the other person comes second (obviously)
                        #  - the age of a0 >= the age of a1, we check for it
                        #  - we don't add any more age constraints; the third person can be another adult of some age; their number is small but not negligible.
                    else:
                        # for m3/m4 we sort by age and sex, this is the only reasonable assumption here
                        # - age of a0 >= age of a1 >= age of a2
                        self.groups[_g]["data"] = Procrustes.stretch_wide_adults(self.groups[_g]["data"].
                                                                                       sort_values(["id_household", "ordinal_person_age", "indicator_person_sex"], ascending=False))
                else:
                    pass # TODO make sure the code is adjusted for mf

            self.groups[_g]["data"] = self.groups[_g]["data"].merge(household_attributes, how="left", on="id_household")
            # we store the household attributes in a separate dataframe and merge it with the transformed one
            # we could do this for transformed tables only, but the performance hit is negligible

    def drop_id_columns(self):
        # we keep this bit in a separate function because we need id_household for children
        for _g in self.subsets:
            _columns_id = [_c for _c in self.groups[_g]["data"] if _c.startswith("id_")]
            self.groups[_g]["data"].drop(columns=_columns_id, inplace=True)

    @staticmethod
    def _get_postfixes(_df) -> list[str] | list[str, ...]:
        _p = [re.search(r"\_a[0-9]$", _c) for _c in _df.columns]
        _p = list(set(_p))
        _p.remove(None)

        if len(_p) > 0:
            return list(set([_match[0] for _match in _p]))
        else:
            return [""]

    @staticmethod # TODO consolidate code
    def _get_postfixes_children(_df) -> list[str] | list[str, ...]:
        _p = [re.search(r"\_c[0-9]$", _c) for _c in _df.columns]
        _p = list(set(_p))
        _p.remove(None)

        if len(_p) > 0:
            return list(set([_match[0] for _match in _p]))
        else:
            return [""]

    def init_models(self, _use_cuda=False, _batch_size=5000, _epochs=5000):
        for _g in self.subsets:
            self.groups[_g]["model"] = CTGANSynthesizer(metadata_constructor(self.groups[_g]["data"], f"{_g}"),
                                                              enforce_rounding=False,
                                                              epochs=_epochs,
                                                              verbose=True,
                                                              cuda=_use_cuda,
                                                              batch_size=_batch_size
                                                              )
            self.groups[_g]["model"].validate(self.groups[_g]["data"])
            self.groups[_g]["model"].auto_assign_transformers(self.groups[_g]["data"])

            for _c, _t in self.groups[_g]["model"].get_transformers().items():
                if _t is not None:
                    _repr = _t.computer_representation
                    self.groups[_g]["model"].update_transformers(column_name_to_transformer={_c: FloatFormatter(enforce_min_max_values=True,
                                                                                                                      computer_representation=_repr)})

    def attach_constraints(self):

        def get_increments(_column_name, _increment):
            # increments: income and minutes
            return {
                'constraint_class': 'FixedIncrements',
                'constraint_parameters': {
                    'column_name': _column_name,
                    'increment_value': _increment
                }
            }

        for _g in self.subsets:
            # these are general constraints for every column of the kind
            # because we use minutes instead of hours we can't say income is always >= time due to minimal rate;
            # we can't also say the opposite due to variable rates
            for _c in self.groups[_g]["data"].columns:
                if "minutes" in _c:
                    self.groups[_g]["model"].add_constraints([get_increments(_c, 15)])
                if "income" in _c:
                    self.groups[_g]["model"].add_constraints([get_increments(_c, 10)])
        # TODO we can have a custom constraint for income rate?

        def get_combinations(_columns: list) -> dict:
            return {
                'constraint_class': 'FixedCombinations',
                'constraint_parameters': {
                    'column_names': _columns
                }
            }

        def get_inequality(_columns: list) -> dict:
            return {
                'constraint_class': 'Inequality',
                'constraint_parameters': {
                    'low_column_name': _columns[0],
                    'high_column_name': _columns[1],
                    'strict_boundaries': False
                }
            }

        for _g in self.subsets:
            postfixes = self._get_postfixes(self.groups[_g]["data"])

            # Age constraints where possible
            if _g.startswith(("c", "m")): # this constraint always works because we sort the data in advance
                if ("ordinal_person_age_a0" in self.groups[_g]["data"].columns) and ("ordinal_person_age_a1" in self.groups[_g]["data"].columns):
                    self.groups[_g]["model"].add_constraints(constraints=[
                        get_inequality(["ordinal_person_age_a1",
                                        "ordinal_person_age_a0"]),
                    ])
            if _g in ["m3", "m4"]:
                if ("ordinal_person_age_a1" in self.groups[_g]["data"].columns) and ("ordinal_person_age_a2" in self.groups[_g]["data"].columns):
                    self.groups[_g]["model"].add_constraints(constraints=[
                        get_inequality(["ordinal_person_age_a2",
                                        "ordinal_person_age_a1"]),
                    ])
            if _g == "m4":
                if ("ordinal_person_age_a2" in self.groups[_g]["data"].columns) and ("ordinal_person_age_a3" in self.groups[_g]["data"].columns):
                    self.groups[_g]["model"].add_constraints(constraints=[
                        get_inequality(["ordinal_person_age_a3",
                                        "ordinal_person_age_a2"]),
                    ])

            if _g.startswith("c") or _g in ["mc3", "mc4"]:
                if ("category_person_legal_marital_status_a0" in self.groups[_g]["data"].columns) and ("category_person_legal_marital_status_a1" in self.groups[_g]["data"].columns):
                    self.groups[_g]["model"].add_constraints(constraints=[
                        # category_person_legal_marital_status in couples should have a fixed number of combinations;
                        # this works for any household with a couple inside by data design
                        get_combinations(["category_person_legal_marital_status_a0",
                                          "category_person_legal_marital_status_a1"])
                    ])

            # add to the list of constraint classes
            for klass in [MetaEmployment, BenefitsIncome, MetaEmploymentNoSecondJob]:
                self.groups[_g]["model"].add_custom_constraint_class(create_custom_constraint_class(klass.is_valid,
                                                                                                          klass.transform,
                                                                                                          klass.reverse_transform),
                                                                           klass.__name__)

            for _p in postfixes:
                # TODO add a check to make sure all columns are in there. should never happen
                # TODO order of constraints matters; do some tests
                # we repeat this for every adult in the household
                quals = [_c for _c in self.groups[_g]["data"].columns if "qualification" in _c and _c.endswith(_p)]
                if len(quals) > 0:
                    self.groups[_g]["model"].add_constraints([
                        get_combinations(quals),
                        # the number of columns can vary from group to group
                    ])
                # FIXME the benefits have been corrupted by imputation and therefore do not pass the constraint check
                if "income_person_second_job" + _p in  self.groups[_g]["data"].columns:
                    self.groups[_g]["model"].add_constraints([
                                            MetaEmployment.get_schema(
                        ["indicator_person_is_self_employed" + _p,
                         "indicator_person_is_employed" + _p,

                         "minutes_person_employment" + _p,
                         "income_person_pay" + _p,
                         "hours_person_overtime" + _p,

                         "hours_person_self_employment" + _p,
                         "income_person_self_employment" + _p,

                         "category_person_job_nssec" + _p,
                         "category_person_job_sic" + _p,

                         "category_person_job_status" + _p,

                         "income_person_second_job" + _p])])
                else:
                    self.groups[_g]["model"].add_constraints([
                                            MetaEmploymentNoSecondJob.get_schema(
                        ["indicator_person_is_self_employed" + _p,
                         "indicator_person_is_employed" + _p,

                         "minutes_person_employment" + _p,
                         "income_person_pay" + _p,
                         "hours_person_overtime" + _p,

                         "hours_person_self_employment" + _p,
                         "income_person_self_employment" + _p,

                         "category_person_job_nssec" + _p,
                         "category_person_job_sic" + _p,

                         "category_person_job_status" + _p])])

    def train(self, save_path, verbose=False):
        for _g in self.subsets:
            if verbose:
                print(_g)
                print(self.groups[_g]["dropouts"])
                print(len(self.groups[_g]["data"]))

            self.groups[_g]["model"].fit(self.groups[_g]["data"])
            self.groups[_g]["model"].save(filepath=save_path + f'model_{_g}.pkl')

            with open(save_path + f'dropouts_{_g}.yaml', 'w') as yml:
                yaml.dump(self.groups[_g]["dropouts"], yml, allow_unicode=True)

    @staticmethod
    def _train_children(_data_household: pd.DataFrame,
                        _max_household_children: int,
                        _household_type: str,
                        _model_location: str):
        """ Trains RF to predict basic attributes of children for a given household type.

        We assume that data has been cleaned and merged together as follows:
        | a0 | ... | c0 | ...

        There is no ids whatsoever since we don't need them for prediction.

        The provided dataset must have at least one child per household with postfixes "_c0" and so on depending on the total number of children.
        Zero-order approximation is to have only attributes of adults for prediction. We select the best of them to predict the age of the oldest child,
        the next step is to add "ordinal_person_age_c0" to the zero-order approximation list of predictors and repeat the procedure of selection but now for the sex of the oldest child.
        We keep going in this pseudo-auto-regressive fashion until we can predict all attributes of the oldest child. If there is, for example, one more kid in the household we use info about adults *and* their older sibling for prediction now.

        Parameters
        ----------
        _data_household
        _max_household_children

        Returns
        -------

        """
        # TODO investigate too young (too old?) parents
        _full_base_predictors = [_c for _c in _data_household.columns if not _c.endswith(tuple(f"_c{_id}" for _id in range(_max_household_children)))]
        _extra_children_predictors = []

        _model_collection = {"children":
                                 {"total": {_max_household_children:
                                                {"id": {_child_id: {"age": tuple(),
                                                                    "sex": tuple(),
                                                                    "ethnic_group": tuple()} for _child_id in range(_max_household_children)}}}}}

        for aux_id in range(_max_household_children):
            _target_map = {
                "age": f"ordinal_person_age_c{aux_id}",
                "sex": f"indicator_person_sex_c{aux_id}",
                "ethnic_group": f"category_person_ethnic_group_c{aux_id}"
            }
            # TODO wait for Python 3.14 to employ string templates

            def _generate_model(_target_code):
                _set = _data_household[_target_map[_target_code]].unique()

                if len(_set) == 1:
                    print(f"Degeneracy at {_household_type} | total children {_max_household_children} | child id {aux_id} in {_target_code}")
                    # degeneracy
                    _model = DummyClassifier(strategy="constant", constant=_set[0])
                    _optimal_predictors = _full_base_predictors

                    _model.fit(_data_household[_optimal_predictors], _data_household[_target_map[_target_code]])
                else:
                    # TODO rf is better than knn in that it can work fine without one hot encoding. still, current implementation does see *all* columns as numeric. works well though. see https://github.com/scikit-learn/scikit-learn/pull/12866
                    _model = RandomForestClassifier(n_estimators=1024,
                                                    criterion='entropy',
                                                    max_depth=None,
                                                    max_features=None,
                                                    random_state=1,
                                                    n_jobs=4,
                                                    )
                    #print(_target_code, _data_household[_target_map[_target_code]].unique())
                    try:
                        x_train, x_test, y_train, y_test = train_test_split(_data_household[_full_base_predictors + _extra_children_predictors],
                                                                            _data_household[_target_map[_target_code]],
                                                                            stratify=_data_household[_target_map[_target_code]],
                                                                            random_state=42, test_size=0.1)
                    except ValueError as e:
                        print(f"Failed with {e}, falling back to non-stratified split")
                        x_train, x_test, y_train, y_test = train_test_split(_data_household[_full_base_predictors + _extra_children_predictors],
                                                                            _data_household[_target_map[_target_code]],
                                                                            random_state=42, test_size=0.1)

                    _optimal_predictors = get_optimal_features(_data_household[_full_base_predictors + _extra_children_predictors],
                                                               x_train, x_test, y_train, y_test)

                    _model.fit(_data_household[_optimal_predictors],
                               _data_household[_target_map[_target_code]])

                    _extra_children_predictors.append(_target_map[_target_code])
                    # only include if the target is not degenerate

                # XXXX _common_path = _model_location + f'{_household_type}_tc{_max_household_children}_cid{aux_id}_{_target_code}.'
                _common_path = os.path.join( _model_location, f'{_household_type}_tc{_max_household_children}_cid{aux_id}_{_target_code}.')
                joblib.dump(_model, _common_path + 'joblib')

                with open(_common_path + 'yaml', 'w') as yml:
                    yaml.dump(_optimal_predictors, yml, allow_unicode=True)

                return _model, _optimal_predictors

            for _k, _v in _target_map.items():
                _model_collection["children"]["total"][_max_household_children]["id"][aux_id][_k] = _generate_model(_k)

        return _model_collection

    def train_children(self, _df_children: pd.DataFrame, save_path: str = "/tmp/", verbose = False):
        # TODO code consolidation
        for _g in self.subsets:
            if _g in ["a1+", "c1", "c2", "c3+"]: # TODO mf
                if verbose:
                    print(f"{_g=}")
                if _g.endswith("+"):
                    _counts = self.groups[_g]["data"]["total_children"].unique()
                    for _total_children in _counts:
                        _df = self.groups[_g]["data"].copy()
                        _df = _df[_df["total_children"].eq(_total_children)].drop(columns="total_children")

                        _columns_id = [_c for _c in _df if _c.startswith("id_")]
                        _columns_id.remove("id_household")
                        _df.drop(columns=_columns_id, inplace=True)

                        if _total_children > MAX_CHILDREN:
                            continue # if too big - skip
                        else:
                            _hh = _df["id_household"].unique()
                            _children = _df_children[_df_children["id_household"].isin(_hh)].copy()

                        if _total_children == 1:
                            _children = _children.rename(columns={_c: _c + "_c0" for _c in _children.columns if not _c.startswith("id_")})
                        else:
                            _children = Procrustes.stretch_wide_children(_children)

                        _df = _df.merge(_children,
                                        how="inner", # inner merge to avoid incomplete records
                                        on="id_household").drop(columns=["id_household"], errors="ignore")

                        if len(_df.drop_duplicates()) == 1:
                            print(f"Extreme degeneracy detected in {_g} with {_total_children} children, can't deal with 1 unique household in a group; dropping them out")
                            self.groups[_g]["data"] = self.groups[_g]["data"][self.groups[_g]["data"]["total_children"].ne(_total_children)]
                            continue

                        comb = _df.apply(pd.unique)
                        exclude = comb[comb.map(len) == 1].map(lambda x: x[0]).to_dict().keys()
                        exclude = [_e for _e in exclude if not _e.endswith(tuple(self._get_postfixes_children(_children)))]
                        # do not drop out degenerate attributes of children
                        _df.drop(columns=exclude, inplace=True)
                        # we do this cleaning because previous data manipulations i.e. merging and such might result in fewer records and therefore lower data variation.

                        #print(_df.apply(pd.unique))
                        _models = self._train_children(_data_household = _df,
                                                       _max_household_children = _total_children,
                                                       _household_type = _g,
                                                       _model_location = save_path)

                        if "children" in self.groups[_g].keys():
                            self.groups[_g]["children"]["total"][_total_children] = _models["children"]["total"][_total_children]
                        else:
                            self.groups[_g]["children"] = _models["children"]

                else:
                    # all households have the same number of children
                    assert "total_children" not in self.groups[_g]["data"].columns

                    _df = self.groups[_g]["data"].copy()

                    _columns_id = [_c for _c in _df if _c.startswith("id_")]
                    _columns_id.remove("id_household")
                    _df.drop(columns=_columns_id, inplace=True)

                    #infer the number of children
                    _total_children = int(''.join(filter(str.isdigit, _g)))

                    if _total_children > MAX_CHILDREN:
                            continue # if too big - skip
                    else:
                        _hh = _df["id_household"].unique()
                        _children = _df_children[_df_children["id_household"].isin(_hh)].copy()

                        if _total_children == 1:
                            _children = _children.rename(columns={_c: _c + "_c0" for _c in _children.columns if not _c.startswith("id_")})
                        else:
                            _children = Procrustes.stretch_wide_children(_children)

                        _df = _df.merge(_children,
                                        how="inner", # inner merge to avoid incomplete records
                                        on="id_household").drop(columns=["id_household"], errors="ignore")

                        if len(_df.drop_duplicates()) == 1:
                            print(f"Extreme degeneracy detected in {_g} with {_total_children} children, can't deal with 1 unique household in a group; dropping them out")
                            self.groups[_g]["data"] = self.groups[_g]["data"][self.groups[_g]["data"]["total_children"].ne(_total_children)]
                            continue

                        comb = _df.apply(pd.unique)
                        exclude = comb[comb.map(len) == 1].map(lambda x: x[0]).to_dict().keys()
                        exclude = [_e for _e in exclude if not _e.endswith(tuple(self._get_postfixes_children(_children)))]
                        # do not drop out degenerate attributes of children
                        _df.drop(columns=exclude, inplace=True)

                        _models = self._train_children(_data_household = _df,
                                                       _max_household_children = _total_children,
                                                       _household_type = _g,
                                                       _model_location = save_path)

                        self.groups[_g]["children"] = _models["children"]

    @staticmethod
    def pad_children(_df: pd.DataFrame) -> pd.DataFrame:
        _personal_indicators = {_c: False for _c in _df.columns if _c.startswith(("mlb", "indicator_person_vocational_qualification", "indicator_person_is"))}

        _income = {_c: 0 for _c in _df.columns if _c.startswith("income")}

        _time = {_c: 0 for _c in _df.columns if _c.startswith("hours", "minutes")}

        _ghq = {_c: 0 for _c in _df.columns if "ghq" in _c}

        _job_status = {_c: 3 for _c in _df.columns if "job_status" in _c}

        _job_class = {_c: 0 for _c in _df.columns if any(x in _c for x in ["nssec", "job_sic"])}

        _marital = {_c: 1 for _c in _df.columns if "marital" in _c}

        _full_qualification = {_c: 96 for _c in _df.columns if "full_highest" in _c}

        _sf12 = {_c: 0 for _c in _df.columns if "_sf" in _c}

        _misc = {_c: 0 for _c in ["ordinal_person_financial_situation", "ordinal_person_life_satisfaction",
                                  "category_person_father_education", "category_person_mother_education"]}
        # NOTE education of parents is tricky as it involves some temporal components
        _df = _df.fillna(value=_personal_indicators | _income | _time | _ghq | _job_status | _job_class | _marital | _full_qualification | _sf12 | _misc)

        _household_attributes = ["category_household_location", "category_household_type",
                                 "ordinal_household_total_cars", "category_household_house_ownership",
                                 "indicator_household_has_central_heating", "total_children"]

        _df[_household_attributes] = _df[_household_attributes].groupby(["id_household"],
                                                                        sort=False)[_household_attributes].apply(lambda x: x.ffill().bfill())
        return _df

    def add_children(self,
            _df: pd.DataFrame,
                         _max_household_children: int,
                         _household_type: str,
                         _model_location: str,
                         _mini_batch_id_: int = None,
                         _micro_batch_id_: int = None,
                         ) -> pd.DataFrame:

        # this is the highest level function
        """Adds children to provided households

        Parameters
        ----------
        _df
        _max_household_children
        _household_type
        _household_location
        _model_location
        _mini_batch_id_
        _micro_batch_id_

        Returns
        -------

        """
        assert _household_type in ["a1+", "c1", "c2", "c3+"]

        # MAKE CHILDREN
        _children_attributes = dict()
        _local_df = _df.copy()

        for aux_id in range(_max_household_children):

            _target_map = {
                    "age": f"ordinal_person_age_c{aux_id}",
                    "sex": f"indicator_person_sex_c{aux_id}",
                    "ethnic_group": f"category_person_ethnic_group_c{aux_id}"
            }

            # loop over target map k, v
            for _target_code, _v in _target_map.items():
                # XXXX_common_path = _model_location + f'{_household_type}_tc{_max_household_children}_cid{aux_id}_{_target_code}.'
                _common_path = os.path.join( _model_location, f'{_household_type}_tc{_max_household_children}_cid{aux_id}_{_target_code}.' )
                # load yaml with optimal predictors
                with open(_common_path + 'yaml', 'r') as stream:
                    _optimal_predictors = yaml.safe_load(stream)

                # check if any of the siblings is there if yes attach corresponding series
                for _c in _optimal_predictors:
                    if _c in _children_attributes.keys():
                        _local_df = pd.concat([_local_df, _children_attributes[_c]], ignore_index=True,  axis=1)

                # load the model itself
                _model = joblib.load(_common_path + 'joblib')
                _children_attributes[_v] = pd.Series(_model.predict(_local_df[_optimal_predictors]), name=_v)
                # predict attributes, store in a series, append to the list

        _children = pd.concat(_children_attributes.values().to_list(), ignore_index=True,  axis=1)
        _children["id_household"] = _df["id_household"]

        if _max_household_children == 1:
            _children = _children.rename(columns={_c: _c[:-3] for _c in _children.columns if not _c.startswith("id_")})
        else:
            _children = Procrustes.stretch_long(_children, "c")
            # TODO code consolidation

        if _household_type != "a1+":
            # for one adult no need to transform the table
            _df = Procrustes.stretch_long(_df, "a")

        _df = pd.concat([_df, _children])

        _df = generate_personal_ids(_df, contains_couples=False if _household_type == "a1+" else True)

        return self.pad_children(_df)

    def generator(self,
                  _local_model: CTGANSynthesizer,
                  _household_type: str,
                  _local_dropouts: dict,
                  _mini_batch_id: int,
                  _micro_batch_id: int,
                  _total_rows: int = 10_000_000,
                  _batch_size: int = 1_000_000,
                  _model_location: str = "/tmp"):

        synthetic_data = _local_model.sample(num_rows=_total_rows, batch_size=_batch_size)
        # assign household ids
        synthetic_data["id_household"] = generate_household_id(_sample_size = len(synthetic_data),
                                                               _mini_batch_id = _mini_batch_id,
                                                               _micro_batch_id = _micro_batch_id,
                                                               _region_id = synthetic_data["category_household_location"],
                                                               _household_type = HOUSEHOLD_ID_MAP[_household_type])

        # reinstate dropouts
        for column_name, column_value in _local_dropouts.items():
            synthetic_data[column_name] = column_value
            # we re-introduce degenerate columns here to make sure all tables have the same structure

        # TODO add columns for Census matching
        # this block covers *all* possible combinations
        if _household_type.startswith("a"):
            if _household_type.endswith("0"):
                # one adult, no children
                synthetic_data = generate_personal_ids(synthetic_data, contains_couples=False)
            else:
                # one adult, several children
                _split = []
                for _children in synthetic_data["total_children"].unique():
                    # loop over children in a1+
                    _split.append(
                        self.add_children(_df=synthetic_data[synthetic_data["total_children"] == _children],
                                      _max_household_children = _children,
                                      _household_type = _household_type,
                                      _model_location = "/tmp", # FIXME no hardcoded values
                                      _mini_batch_id_ = _mini_batch_id,
                                      _micro_batch_id_ = _micro_batch_id)
                    )
                synthetic_data = pd.concat(_split)

        elif _household_type.startswith("c"):
            if _household_type.endswith("0"):
                # couple, no children
                synthetic_data = generate_personal_ids(synthetic_data, contains_couples=True)
                synthetic_data = Procrustes.stretch_long(synthetic_data, "a")
            else:
                # couple, several children
                if _household_type in ["c1", "c2"]:
                    synthetic_data = self.add_children(_df=synthetic_data,
                                      _max_household_children = int(''.join(filter(str.isdigit, _household_type))),
                                      _household_type = _household_type,
                                      _model_location = "/tmp", # FIXME no hardcoded values
                                      _mini_batch_id_ = _mini_batch_id,
                                      _micro_batch_id_ = _micro_batch_id)
                else:
                    # loop over children c3+
                    _split = []
                    for _children in synthetic_data["total_children"].unique():
                        # loop over children in a1+
                        _split.append(
                            self.add_children(_df=synthetic_data[synthetic_data["total_children"] == _children],
                                          _max_household_children = _children,
                                          _household_type = _household_type,
                                          _model_location = "/tmp", # FIXME no hardcoded values
                                          _mini_batch_id_ = _mini_batch_id,
                                          _micro_batch_id_ = _micro_batch_id)
                        )
                    synthetic_data = pd.concat(_split)

        elif _household_type.startswith("mc"):
            # couple + some other people
            synthetic_data = generate_personal_ids(synthetic_data, contains_couples=True)
        else: # TODO this only works for m3, m4 not mf
            synthetic_data = generate_personal_ids(synthetic_data, contains_couples=False)

        return synthetic_data

    @staticmethod
    def assign_matching_columns(_data):
        pass
        #TODO finish this one
