# HR 13/03/25 Getting pre-processing running from command line

from synthwave.utils.uk.understanding_society import preprocess_usoc_data
import argparse

def do_it(_path):
    try:
        print('Trying to get pre-processed ind and hh pickle files...')
        preprocess_usoc_data(_path, skip_conversion=True)
        print('Done!')
    except:
        print("Exception occurred, probably because the pickle files weren't found; running pre-processing step and caching pickles... ")
        preprocess_usoc_data(_path, skip_conversion=False)
        print('Done!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="Data source path")

    args = parser.parse_args()
    us_path = args.p
    do_it(us_path)
