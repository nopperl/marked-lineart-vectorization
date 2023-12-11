#!/usr/bin/env python3
from argparse import ArgumentParser
from glob import glob
from random import seed, shuffle
from os import makedirs
from os.path import basename, normpath, join, split
from shutil import copy, move


def combine_data(dirs, output_dir="data/processed"):
    combined_output_dir = join(output_dir, "-".join(("-".join(basename(normpath(d)).split("-")[:2]) for d in dirs)))
    if len(basename(dirs[0]).split("-")) > 1:
        combined_output_dir += "-" + "-".join(basename(normpath(dirs[0])).split("-")[2:])
    makedirs(combined_output_dir, exist_ok=True)
    print("Save to " + combined_output_dir)
    for input_dir in dirs:
        for filename in glob(join(input_dir, "*.svg")) + glob(join(input_dir, "*.png")):
            subset_name = "-".join(basename(normpath(input_dir)).split("-")[:2])
            new_name = subset_name + "_" + basename(filename)
            copy(filename, join(combined_output_dir, new_name))
    return combined_output_dir
    

def split_data(data_dir, test_subset="tonari", val_split=0.1, random_seed=1234):
    train_dir = join(data_dir, "train/images")
    validation_dir = join(data_dir, "validation/images")
    test_dir = join(data_dir, "test/images")
    makedirs(train_dir, exist_ok=True)
    makedirs(validation_dir, exist_ok=True)
    makedirs(test_dir, exist_ok=True)

    if test_subset in data_dir:
        if test_subset == "tonari":
            test_svgs = glob(join(data_dir, "tonari-black*4?*.svg")) \
                        + glob(join(data_dir, "tonari-blue*4?*.svg")) \
                        + glob(join(data_dir, "tonari-red*4?*.svg")) \
                        + glob(join(data_dir, "tonari-lime*4?*.svg"))
        elif test_subset == "sketchbench":
            test_svgs = glob(join(data_dir, "sketchbench-black_Art_freeform_A*.svg"))
        else:
            raise ValueError("Invalid test_subset specified")
        for f in test_svgs:
            if test_subset == "tonari" and ("DOU" in f or "5" in f or "6" in f):  # take too long to test
                continue
            move(f, test_dir)
            move(f.replace(".svg", ".png"), test_dir)
    
    seed(random_seed)
    filenames = glob(join(data_dir, "tonari-*.svg")) + glob(join(data_dir, "sketchbench*.svg"))
    shuffle(filenames)
    limit = int(len(filenames) * val_split)
    validation_files = filenames[:limit]
    for f in validation_files:
        move(f, validation_dir)
        move(f.replace(".svg", ".png"), validation_dir)
    for f in glob(join(data_dir, "*.*")):
        move(f, train_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dirs", nargs="+", default=["data/processed/tonari-black-512-0.512", "data/processed/tonari-blue-512-0.512", "data/processed/tonari-red-512-0.512","data/processed/tonari-lime-512-0.512", "data/processed/sketchbench-black-512-0.512", "data/processed/tuberlin-black-512-0.512"], help="The directories corresponding to data subsets to combine to a complete dataset")
    parser.add_argument("-o", "--output_dir", default="data/processed")
    parser.add_argument("-v", "--val_split", default=0.1, help="Train/validation split percentage")
    parser.add_argument("-s", "--seed", default=1234)
    parser.add_argument("-t", "--test_subset", default="tonari")
    args = parser.parse_args()
    if test_subset == "sketchbench":
        output_dir = join(args.output_dir, test_subset)
    combined_output_dir = combine_data(args.input_dirs, output_dir=output_dir)
    split_data(combined_output_dir, test_subset=args.test_subset, val_split=args.val_split, random_seed=args.seed)
