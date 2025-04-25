# HR 15/03/25 Correct and train imputed data

import os
import pandas as pd
from synthwave.synthesizer.postimputation.correction import correct_imputed_data
from synthwave.synthesizer.uk.generator import Syntets
import argparse


def main(data_path, save_path=None):

    if not save_path:
        save_path = os.path.join(data_path, "synthwave", "trained")

    # 1. Correct imputed data
    print("Correcting imputed data...")
    adults = pd.read_csv(os.path.join(data_path, "synthwave", "imputed", "imputed_data.csv"), dtype_backend="pyarrow")
    adults = correct_imputed_data(adults)
    print("Done!")

    # 2. Train model
    print("Creating generator and restructuring data...")
    generator = Syntets(adults)
    generator.split_data()
    generator.restructure_data()
    print("Done!")

    # load dataset
    print("Tidying up child data...")
    children = pd.read_parquet(os.path.join(data_path, "children_non_imputed_middle_fidelity.parquet")).drop(columns=["id_person"])

    # convert data types
    children[["ordinal_person_age", "category_person_ethnic_group"]] = children[["ordinal_person_age", "category_person_ethnic_group"]].astype("uint8[pyarrow]")

    # drop households with incomplete records
    crooked_records = pd.unique(children[children["category_person_ethnic_group"].isna()]["id_household"])
    children = children[~children["id_household"].isin(crooked_records)]  # NOTE do not drop duplicates ever, this destroys twins
    print("Done!")

    print("Training child data...")
    # children = children.sample(frac=2.0, replace=True)
    generator.train_children(children, verbose=True)
    print("Done!")

    generator.drop_id_columns()  # we need ids to learn how children are formed
    generator.locate_degenerate_distributions()
    generator.convert_types()
    generator.init_models(_epochs=1)
    generator.attach_constraints()

    print("Running main training...")
    generator.train(save_path=save_path)
    print("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="Data source path")

    args = parser.parse_args()
    data_path = args.p
    main(data_path)
