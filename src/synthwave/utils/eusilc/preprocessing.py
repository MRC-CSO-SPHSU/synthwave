import warnings

from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from synthwave.utils.general import scale_sample

from importlib.resources import files

MAPPER = yaml.safe_load(files("synthwave.data.eusilc").joinpath('maps.yaml').read_text())

RLM = yaml.safe_load(files("synthwave.data.eusilc").joinpath('relationship_maps.yaml').read_text())

metadata = dict((("D", yaml.safe_load(files("synthwave.data.eusilc").joinpath('household_register.yaml').read_text())),
                 ("R", yaml.safe_load(files("synthwave.data.eusilc").joinpath('personal_register.yaml').read_text())),
                 ("H", yaml.safe_load(files("synthwave.data.eusilc").joinpath('household_data.yaml').read_text())),
                 ("P", yaml.safe_load(files("synthwave.data.eusilc").joinpath('personal_data.yaml').read_text()))
                 ))

def generate_names(metadata_, targets_):
    for _name in ["D", "R", "H", "P"]:
        for k_ in targets_[_name].keys():
            prefix_ = metadata_[_name]["Values"][k_]["Type"]
            targets_[_name][k_] = "_".join([prefix_, targets_[_name][k_]])
    return targets_

TARGET_SYNTHETIC_COLUMNS = generate_names(metadata, yaml.safe_load(files("synthwave.data.eusilc").joinpath('target_synthetic_columns.yaml').read_text()))

def read_full_dataset(path: Path, country: str):
    return {table_type: pd.read_csv(path / f'UDB_c{country}22{table_type}.csv') for table_type in ["D", "R", "H", "P"]}
# FIXME we cant that easy discard all that na records; no
#  we need extra metadata explaining what strata this can't be applied to (say younger than 16)
#  and how to mitigate this. some can be achieved by padding and further logical constraints some - not

def get_birth_location(_record):
    match _record:
        case "LOC": return True
        case "OTH": return False
        case "EU": return False
        case _: return pd.NA

def relator(df_):
    return tuple(sorted(pd.unique(df_.fillna(0).values.ravel())))
# zero is a self-reference

def _cleaner(rels):
    if len(rels) == 1 and rels[0] == 0:
        return tuple()
    else:
        return tuple(sorted([int(i) for i in rels if i != 0]))

def acceptor(rels):
    if len(rels) > 0:
        for i in RLM["FORBIDDEN_RELATIONSHIPS"]:
            if i in rels:
                return False
    return True

def padder(data_, hh_size, mask_):
    # for a singular household only!
    for id_ in range(1, hh_size + 1):
        data_.loc[mask_ & data_["id_internal"].eq(id_), f"RG_{id_}"] = 0
        # pad the main diagonal with zeroes, that's a new code for self-reference
    return data_

class Validator:
    # methods to ensure all variables are sensible i.e., within their respective bounds and not null
    @staticmethod
    def is_degenerate(df_, dataset_name=None):
        to_drop = []
        # checks if there is only one non-null value in the column
        # we assume no null columns here
        for cl in df_.columns:
            if str(df_[cl].dtype) == 'halffloat[pyarrow]': # FIXME wait for next arrow release
                print("XXX")
                continue

            t = pd.unique(df_[cl])
            if (len(t) == 1) or ((len(t) == 2) and pd.isna(t).any()):
                if (dataset_name is None) or (cl.startswith("RG_")) or (cl.endswith("_F")) or (cl.endswith("_IF")):
                    _name = ''
                else:
                    _name = metadata[dataset_name]['Values'][cl]['Name']

                if not cl.startswith("RG_"): # a fix to keep the relationships in place
                    _msg = f"Only one unique non-null value in {cl}: {_name}"
                    warnings.warn(_msg) # to get properly formatted string
                    to_drop.append(cl)

        return to_drop

    @staticmethod
    def is_null(df_, dataset_name=None):
        # for empty columns
        to_drop = []
        for cl in df_.columns:
            if df_[cl].isna().all():
                if (dataset_name is None) or (cl.startswith("RG_")):
                    _name = ''
                else:
                    _name = metadata[dataset_name]['Values'][cl]['Name']
                warnings.warn(f"{cl}: {_name} is null")
                to_drop.append(cl)

        #df_.drop(columns=to_drop, inplace=True)
        return to_drop

    @staticmethod
    def is_flag(df_):
        flags = []
        for cl in df_.columns:
            if cl.endswith("_F") or cl.endswith("_IF"):
                flags.append(cl)
        return flags

# get columns from metadata
# get columns from file
# find difference from both sides


# It is impossible to establish certain relationships within the household, therefore we ignore them
# such RG_# values are empty, and the corresponding flag is -1.
# we seek for such flag in every RG_#_F column, and select relevant households only#

def preprocess_eusilc(data_path, country):
    data = read_full_dataset(data_path, country)
    rl_indicators = []
    for c in data["R"].columns:
        if c.startswith("RG_") and c.endswith("F"):
            rl_indicators.append(c)

    corrupted_households = data["R"].copy()[["RX030"] + rl_indicators]
    assert ~corrupted_households[rl_indicators].isna().any().any()
    # TODO infer?
    corrupted_households[rl_indicators] = corrupted_households[rl_indicators].eq(-1)
    corrupted_households["indicator"] = corrupted_households[rl_indicators].sum(1)
    corrupted_households = (corrupted_households.
                               drop(columns=rl_indicators).
                               groupby(["RX030"]).
                               sum().
                               reset_index())
    corrupted_households = corrupted_households[corrupted_households["indicator"] > 0]["RX030"]
    corrupted_individuals = data["R"][data["R"]["RX030"].isin(corrupted_households)]["RB030"]

    data["R"] = data["R"][~data["R"]["RX030"].isin(corrupted_households)]
    data["D"] = data["D"][~data["D"]["DB030"].isin(corrupted_households)]
    data["H"] = data["H"][~data["H"]["HB030"].isin(corrupted_households)]

    data["P"] = data["P"][~data["P"]["PB030"].isin(corrupted_individuals)]

    _list = []
    for c in data["R"].columns:
        if c.startswith("RG_") and c.endswith("F"):
            _list.append(c)

    data["R"] = data["R"].drop(columns=_list)
    # we remove relationship flags since they do not add any useful information to our data; all `bad` households have been removed already

    # here, we drop columns that are of no use to us at all meaning they are not listed in the target file
    for t_ in data.keys():
        _columns = TARGET_SYNTHETIC_COLUMNS[t_]
        _to_keep = []
        for _cc in data[t_].columns:
            if _cc.startswith("RG_"):
                _to_keep.append(_cc)
                # by this time there is no RG_#_F
        for _k in _columns.keys():
            _to_keep.append(_k)
            # the default state is that the variable is flagged
            if "Extra" in metadata[t_]["Values"][_k].keys():
                if not ("NotFlagged" in metadata[t_]["Values"][_k]["Extra"]):
                    _to_keep.append(_k + "_F")
            else:
                _to_keep.append(_k + "_F")
                # income is always double-flagged
                if metadata[t_]["Values"][_k]["Type"] == "income":
                    _to_keep.append(_k + "_IF")
        data[t_] = data[t_][_to_keep]


    for _dataset in data.keys():
        _to_drop = Validator.is_null(data[_dataset], _dataset)
        data[_dataset] = data[_dataset].drop(columns=_to_drop)
    # drop columns that are all null

    for _dataset in data.keys():
        for cl in metadata[_dataset]["Values"].keys():
            if cl in data[_dataset].columns:
                match metadata[_dataset]["Values"][cl]["Type"]:
                    case "id":
                        data[_dataset][cl] = data[_dataset][cl].astype("uint64[pyarrow]")
                    case "category":
                        if "DataType" in metadata[_dataset]["Values"][cl].keys():
                            data[_dataset][cl] = data[_dataset][cl].astype(metadata[_dataset]["Values"][cl]["DataType"])
                        else:
                            data[_dataset][cl] = data[_dataset][cl].astype("category")
                    case "income":
                        if "Range" in metadata[_dataset]["Values"][cl]:
                            if metadata[_dataset]["Values"][cl]["Range"]["Min"] < 0:
                                data[_dataset][cl] = round(data[_dataset][cl]).astype("int32[pyarrow]")
                            else:
                                data[_dataset][cl] = round(data[_dataset][cl]).astype("uint32[pyarrow]")
                        else:
                            data[_dataset][cl] = round(data[_dataset][cl]).astype("uint32[pyarrow]")
                    case "indicator":
                        data[_dataset][cl] = data[_dataset][cl].replace({1: True, 2: False}).astype("bool[pyarrow]")
                    case "ordinal":
                        data[_dataset][cl] = data[_dataset][cl].astype(metadata[_dataset]["Values"][cl]["DataType"])
                    case "weight":
                        data[_dataset][cl] = data[_dataset][cl].astype("float32[pyarrow]")
                    case _:
                        if "Name" in metadata[_dataset]["Values"][cl]:
                            print(_dataset, cl, metadata[_dataset]["Values"][cl])
                        else:
                            print(_dataset, cl)
    # do some manual memory management


    for _dataset in data.keys():
        _to_drop = Validator.is_degenerate(data[_dataset], _dataset)
        data[_dataset] = data[_dataset].drop(columns=_to_drop)
    # remove columns that contain only one value or one value and null

    for _dataset in data.keys():
        for name_ in MAPPER.keys():
            if name_.startswith(_dataset) and name_ in data[_dataset].columns:
                data[_dataset][name_] = data[_dataset][name_].replace(MAPPER[name_])


    present_columns = dict((k, data[k].columns) for k in data.keys())


    for _dataset in data.keys():
        design_columns = set()

        # loop over all column descriptions in metadata and make sure only those selected for dissemination are being picked
        for name_ in metadata[_dataset]["Values"].keys():
            if not (("Extra" in metadata[_dataset]["Values"][name_].keys()) and (["NotDisseminated"] in metadata[_dataset]["Values"][name_]["Extra"])):
                design_columns.add(name_)

        _to_add = set()
        # extract *all* column names from metadata
        for name_ in design_columns:
            if metadata[_dataset]["Values"][name_]["Type"] == "income" and "HY150" not in name_:
                _to_add.add(f"{name_}_IF")

            if "Extra" not in metadata[_dataset]["Values"][name_]: # no extra conditions whatsoever
                _to_add.add(f"{name_}_F")
            else:
                if "NotFlagged" not in metadata[_dataset]["Values"][name_]["Extra"]:
                    _to_add.add(f"{name_}_F")

        design_columns = design_columns | _to_add
        # let's say by design should be 4 columns: "a", "b", "c", "d".
        # The file however only contains "b", "c", "d", and "e".
        # The first part of this section finds the intersection of these two sets and removes it from the first one leaving only "a".
        # This column is supposed to be in the file but not there therefore we warn the user and drop it
        # the rest of the code does the opposite. it finds columns that are in the file
        # i.e. "e" but not in the metadata and lets the user know
        design_missing = design_columns - set(present_columns[_dataset])

        if len(design_missing) != 0:
            for f in design_missing:

                if f in metadata[_dataset]['Values'].keys():
                    if "Name" in metadata[_dataset]['Values'][f].keys():
                        _stub = f"{f}: {metadata[_dataset]['Values'][f]['Name']} "
                    else:
                        _stub = f"{f}: "
                else:
                    _stub = f"{f}: is likely a flag column and it "
                warnings.warn(_stub + "is missing;  it won't be used in any future endeavours.")


        undocumented_columns = set(present_columns[_dataset]) - design_columns
        if len(undocumented_columns) != 0:
            for f in undocumented_columns:
                warnings.warn(f"{f}: is undocumented;"
                              f" it won't be used in any future endeavours.")

        data[_dataset] = data[_dataset][sorted(list(design_columns & set(present_columns[_dataset])))]
        #print(design_columns & set(present_columns[_dataset])) # intersection of two sets
        # TODO now we drop out a lot of columns before this stage; this bit either needs to be run earlier, we could also redesign it

    for _dataset in data.keys():
        _not_needed = []
        for _cl in data[_dataset].columns:
            if _cl.endswith("_IF") or (_cl.endswith("_F") and ("Y" in _cl)):
                _not_needed.append(_cl) # disregard them at the moment
            else:
                if _cl.endswith("_F"):
                    if not pd.Series(pd.unique(data[_dataset][_cl])).lt(-1).any():
                        # -1 is the default value for missing data, other are special codes
                        _not_needed.append(_cl)
        if len(_not_needed) > 0:
            data[_dataset] = data[_dataset].drop(columns=_not_needed)


    data["R"] = data["R"].drop(columns=["RB220_F", "RB230_F", "RB240_F"])
    # we also need no flags for missing ids.

    for _dataset in data.keys():
        data[_dataset] = data[_dataset].rename(columns=TARGET_SYNTHETIC_COLUMNS[_dataset])
    # renaming columns goes right before the last merge, not anywhere else

    full_data = (data["R"].
                 merge(data["P"], how="left", on=["id_household", "id_person"]). # left join since register is more complete
                 merge(data["D"], how="left", on=["id_household"]).
                 merge(data["H"], how="left", on=["id_household"]))

    #FIXME bloody HH030 and similar are not regular integers

    full_data["category_occupation_job"] = full_data["category_occupation_job"].map(lambda x: x // 10 if x >= 10 else x).astype("uint8[pyarrow]")
    # some codes are one digit only but must be two, we introduce a fix here

    full_data.loc[full_data["PL051A_F"].isna(), "category_occupation_job"] = 11 # children aged 15 and below now have their own category
    full_data.loc[full_data["PL051A_F"].eq(-2), "category_occupation_job"] = 10 # aged 16 and above but not formally employed
    # zero is reserved for armed forces
    full_data = full_data.drop(columns="PL051A_F") # drop the flag now, all n/a values are missing by reduction since we've excluded all other options



    full_data["indicator_is_local_born"] = full_data["category_country_birth"].map(get_birth_location).astype("bool[pyarrow]")
    full_data = full_data.drop(columns="category_country_birth")

    full_data.loc[full_data["PL060_F"].isna(), "ordinal_hours"] = 0 # children work zero hours by definition
    full_data.loc[full_data["PL060_F"].eq(-2), "ordinal_hours"] = 0 # those who have no job have no hours

    # other possible codes are -1 and -6, both will be imputed
    full_data = full_data.drop(columns="PL060_F")

    # this one is a bit tricky because we change the semantics of the actual question
    full_data.loc[full_data["PL145_F"].isna(), "indicator_full_time"] = False
    full_data.loc[full_data["PL145_F"].eq(-2), "indicator_full_time"] = False

    full_data = full_data.drop(columns="PL145_F")

    full_data["category_region"] = full_data["category_region"].str.extract(r'(\d+)').astype("uint8[pyarrow]")
    # one country only, 1 digit per region so this approach works

    full_data.loc[full_data["ordinal_age"].le(15), "income_old_age_benefits"] = 0
    full_data.loc[full_data["ordinal_age"].le(15), "income_self_employment"] = 0
    full_data.loc[full_data["ordinal_age"].le(15), "income_non_cash"] = 0
    full_data.loc[full_data["ordinal_age"].le(15), "income_gross_salary"] = 0
    # for kids these are zero by definition

    # these attributes can't be imputed for children
    # ordinal_life_satisfaction
    # category_read_books
    # bmi
    # limitations health
    # long term sick
    # ordinal health
    #
    # TODO education level for those 15 and below can be in theory propagated, but I dont' know how yet
    #  indicator is student is the same
    #  marital status/ consensual union can be propagated?

    full_data["indicator_is_selected"] = pd.NA
    full_data["indicator_is_selected"] = full_data["indicator_is_selected"].astype("bool[pyarrow]")

    check_ = full_data[["id_household", "ordinal_household_size"]].drop_duplicates().merge(full_data.groupby(["id_household"]).size().to_frame("calculated_total").reset_index(), on="id_household")
    ((check_["ordinal_household_size"] - check_["calculated_total"]) == 0 ).all()
    # TODO write a sanitizing routine
    #  all household sizes are valid

    full_data["household_type"] = pd.NA

    full_data.loc[full_data["ordinal_household_size"].eq(1), "indicator_is_selected"] = True
    full_data.loc[full_data["ordinal_household_size"].eq(1), "household_type"] = "sa"

    relationships = []
    for c in full_data.columns:
        if "RG_" in c and not c.endswith("F"):
            relationships.append(c)

    family_structure = full_data.copy()[["id_household"] + relationships]

    households = (family_structure.
                  groupby(["id_household"]).
                  apply(lambda x: relator(x), include_groups=False).
                  to_frame("existing_relationships").
                  reset_index())

    households["existing_relationships"] = households["existing_relationships"].map(_cleaner)
    households = households.merge(family_structure.groupby(["id_household"]).size().to_frame("total_individuals").reset_index(), on="id_household")

    households = households[households["existing_relationships"].map(acceptor)]



    full_data = full_data[full_data["id_household"].isin(households["id_household"])]

    for hh_ in tqdm(households["id_household"]):
        household_indicator_ = full_data["id_household"].eq(hh_)
        full_data = padder(full_data,
                           households[households["id_household"].eq(hh_)]["total_individuals"].values[0],
                           household_indicator_)

    full_data.loc[:,"total_adults"] = 1
    full_data.loc[:,"total_children"] = 0
    # by design now every household has at least 1 adult and at least 0 children

    hh_ids = pd.unique(full_data[full_data["id_person"].isin(full_data["id_partner"])]["id_household"])

    (full_data.loc[full_data["id_household"].isin(hh_ids) &
                  full_data["category_household_type"].isin([6, 7]) &
                  full_data["category_household_type2"].eq(4), "ordinal_household_size"] == 2).all()

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].isin([6, 7]) & full_data["category_household_type2"].eq(4)

    full_data.loc[_mask, "indicator_is_selected"] = True
    full_data.loc[_mask, "household_type"] = "cl"
    full_data.loc[_mask, "total_adults"] = 2

    full_data.loc[full_data["id_household"].isin(hh_ids) &
                  full_data["category_household_type"].eq(8) &
                  full_data["category_household_type2"].isin([4, 5, 6]), "indicator_is_selected"] = False

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].eq(10) & full_data["category_household_type2"].eq(5)
    full_data.loc[_mask, "indicator_is_selected"] = True
    full_data.loc[_mask, "household_type"] = "c1"
    full_data.loc[_mask, "total_adults"] = 2
    full_data.loc[_mask, "total_children"] = 1

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].eq(11) & full_data["category_household_type2"].eq(5)
    full_data.loc[_mask, "indicator_is_selected"] = True
    full_data.loc[_mask, "household_type"] = "c2"
    full_data.loc[_mask, "total_adults"] = 2
    full_data.loc[_mask, "total_children"] = 2

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].eq(12) & full_data["category_household_type2"].eq(5)
    full_data.loc[_mask, "indicator_is_selected"] = True
    full_data.loc[_mask, "total_adults"] = 2
    full_data.loc[_mask, "total_children"] = full_data.loc[_mask, "ordinal_household_size"] - full_data.loc[_mask, "total_adults"]
    full_data.loc[_mask, "household_type"] = full_data.loc[_mask, "total_children"].astype(str).map(lambda x: f"c{x}")

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].isin([13, 16]) & full_data["category_household_type2"].eq(5)
    full_data.loc[_mask, "indicator_is_selected"] = False
    # TODO this is not a universal approach, we desperately need some relationship graphs

    hh_ids = pd.unique(full_data[full_data["indicator_is_selected"].isna()]["id_household"])

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].isin([6, 7, 8]) & full_data["category_household_type2"].isin([2, 3, 7])
    full_data.loc[_mask, "indicator_is_selected"] = False

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].eq(9) & full_data["category_household_type2"].eq(7)
    full_data.loc[_mask, "indicator_is_selected"] = False

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].eq(9) & full_data["category_household_type2"].eq(2)
    full_data.loc[_mask, "indicator_is_selected"] = True
    full_data.loc[_mask, "total_adults"] = 1
    full_data.loc[_mask, "total_children"] = full_data.loc[_mask, "ordinal_household_size"] - full_data.loc[_mask, "total_adults"]
    full_data.loc[_mask, "household_type"] = full_data.loc[_mask, "total_children"].astype(str).map(lambda x: f"s{x}")

    _mask = full_data["id_household"].isin(hh_ids) & full_data["category_household_type"].isin([10, 11, 13, 16])
    full_data.loc[_mask, "indicator_is_selected"] = False

    full_data = full_data[full_data["indicator_is_selected"]].drop(columns=["indicator_is_selected"])

    full_data.loc[full_data["category_household_type"] == 13, "category_household_type"] = 5
    # TODO this is a one-time fix for households comprised of a single child who's an adult at the same time

    full_data["indicator_is_adult"] = False
    full_data["indicator_is_adult"] = full_data["indicator_is_adult"].astype("bool[pyarrow]")

    full_data.loc[full_data["category_household_type"].eq(5), "indicator_is_adult"] = True

    full_data.loc[full_data["id_person"].isin(full_data["id_partner"]), "indicator_is_adult"] = True

    parent_ids = pd.unique(pd.concat([full_data["id_father"].dropna(), full_data["id_mother"].dropna()]))

    full_data.loc[full_data["id_person"].isin(parent_ids), "indicator_is_adult"] = True

    # seems to be a good spot to split into children and adults
    redundant_columns = []
    for c_ in full_data.columns:
        if "RG_" in c_:
            redundant_columns.append(c_)

    full_data = full_data.drop(columns=redundant_columns)

    target_adults = full_data[full_data["indicator_is_adult"]].copy()

    target_adults = target_adults.drop(columns=["id_father", "id_mother", "indicator_is_adult", "household_type", "category_household_type",
                                                "category_household_type2", "ordinal_household_size", "id_internal", "indicator_is_resident"], errors="ignore")

    target_adults["weight_household_cross_section"] = round(target_adults["weight_household_cross_section"]).astype("uint16[pyarrow]")

    target_adults["ordinal_number_of_rooms_x10"] = round(target_adults["ordinal_number_of_rooms"].astype("float[pyarrow]") * 10).astype("uint8[pyarrow]")
    target_adults = target_adults.drop(columns=["ordinal_number_of_rooms"])
    #FIXME this is a temporary fix because R can't read half-floats at the moment

    target_adults = scale_sample(target_adults, "weight_household_cross_section")

    _propagation_mask = target_adults["category_job_status"].isin([2, 3, 4, 5, 6, 7, 8]) & target_adults["income_gross_salary"].eq(0) & target_adults["income_self_employment"].eq(0)

    target_adults[_propagation_mask][["income_non_cash", "income_gross_salary", "income_self_employment", "ordinal_hours", "indicator_full_time", "category_job_status"]].value_counts(dropna=False)
    # FIXME too loose condition, add income_non_cash == 0
    #  this is also out best guess, some of them might be, for example, se with no current income

    target_adults.loc[_propagation_mask, "ordinal_hours"] = 0
    target_adults.loc[_propagation_mask, "indicator_full_time"] = False

    # TODO see how imputation goes, introduce employed/self-employed flags if it doesn't work

    children = full_data[~full_data["indicator_is_adult"]].copy().reset_index(drop=True)

    children = children[["ordinal_age", "indicator_sex", "id_household", "indicator_is_local_born" ]]
    # NOTE do not scale children, there is no need for that
    return target_adults, children
