#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os.path import join, exists
from matplotlib import pyplot as plt
from json import dump
from scipy.stats import iqr

def extract_tags(log_dir, tag, filename):
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()
    events = event_accumulator.Scalars(tag)
    data = [(event.step, event.value) for event in events]

    df = pd.DataFrame(data, columns=["step", tag])
    df.set_index("step", inplace=True)
    df.to_csv(filename)
    return df

def extract_image(log_dir, image_tag, filename):
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()
    events = event_accumulator.Images(image_tag)
    image = events[-1].encoded_image_string
    with open(filename, "wb") as image_file:
        image_file.write(image)


def plot(df, filename, window):
    smoothed = df.iloc[:, 0].rolling(window).mean()
    plt.plot(df.index, smoothed, color="black")
    plt.xlabel("step")
    plt.ylabel(df.columns[0])
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to extract TensorBoard event data for multiple tags.")
    parser.add_argument("--log_dir", required=True, help="Path to the TensorBoard event logs directory")
    parser.add_argument("--tags", nargs="*", default=["train_loss", "val_loss", "train_iou_step", "val_iou_step"], help="Tags to extract from event logs")
    parser.add_argument("--image_tags", nargs="*", default=["generated_images", "generated_images_train", "real_images", "real_images_train"], help="Image tags to extract from event logs")
    parser.add_argument("--step_limit", type=int, default=100000, help="The maximum number of steps displayed in the graph")
    parser.add_argument("--window", type=int, default=30, help="The maximum number of steps displayed in the graph")
    args = parser.parse_args()
    summary = {}
    for tag in args.tags:
        cache_filename = join(args.log_dir, tag + ".csv")
        if not exists(cache_filename):
            df = extract_tags(args.log_dir, tag, cache_filename)
        else: 
            df = pd.read_csv(cache_filename, index_col="step")
        plot_filename = join(args.log_dir, tag + ".pdf")
        df = df[df.index <= args.step_limit]
        plot(df, plot_filename, args.window)
        last_window = df.iloc[-args.window:, 0].tolist()
        summary[tag] = {"values": last_window, "median": np.median(last_window), "iqr": iqr(last_window)}
    summary_filename = join(args.log_dir, "summary.json")
    with open(summary_filename, "w") as summary_file:
        dump(summary, summary_file)
    for image_tag in args.image_tags:
        image_filename = join(args.log_dir, image_tag + ".png")
        extract_image(args.log_dir, image_tag, image_filename) 

