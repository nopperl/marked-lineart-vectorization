#!/usr/bin/env python


from os import makedirs
from os.path import exists, join
from glob import glob
from itertools import chain
from json import dump, load
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from svgelements import Color
from svgpathtools import svg2paths2
from scipy.stats import iqr

plt.rcParams.update({'font.size': 14})


datasets = ["tonari", "sketchbench", "tuberlin"]
data_dir = "../data/clean"
report = "../report"
figures = join(report, "figures")
tables = join(report, "tables")

makedirs(report, exist_ok=True)
makedirs(figures, exist_ok=True)
makedirs(tables, exist_ok=True)


def one_is_endpoint(intersection, tol=.1):
    i = intersection
    return i[0][0] < tol or i[0][0] > 1 - tol \
        or i[1][0] < tol or i[1][0] > 1 - tol \

def calculate_intersections(paths, path_attributes=None, ignore_color : Optional[str] =None):
    no_overlapping = 0
    check_color = path_attributes is not None and ignore_color is not None
    for i, path1 in enumerate(paths):
        # Skip white patches
        if check_color and Color(path_attributes[i].get("stroke")) == Color(ignore_color):
            continue
        for j in range(i + 1, len(paths)):
            if check_color and Color(path_attributes[j].get("stroke")) == Color(ignore_color):
                continue
            path2 = paths[j]
            if path1 == path2:
                no_overlapping += 1
            try:
                intersections = path1.intersect(path2)
            except (AssertionError, ValueError):
                continue
            if len(intersections) > 1 and not all_endpoint:
                all_endpoint = all(one_is_endpoint(i) for i in intersections)
                if not all_endpoint:
                    no_overlapping += 1
    return no_overlapping

def get_statistics(data_dir, dataset, calculate_overlapping=False):
    data_dir = join(data_dir, dataset)
    cache_filename = data_dir + ".json"
    if exists(cache_filename):
        with open(cache_filename, "r") as cache_file:
            return load(cache_file)
    svg_files = Path(data_dir).rglob("*.svg")
    stats = {
        "paths": [],
        "curves": [],
        "average curves per path": [],
        "average path length": [],
        "average curve length": [],
    }
    if calculate_overlapping:
        stats.update({"overlapping curves": []})
    for svg_file in svg_files:
        paths, path_attributes, _ = svg2paths2(svg_file)
        no_paths = len(paths)
        curves = list(chain(*((curve for curve in path) for path in paths)))
        no_curves = len(curves)
        avg_curve_len = np.median([curve.length() for curve in curves])
        avg_path_len = np.median([path.length() for path in paths])
        avg_curve_per_path = np.median([len(path) for path in paths])
        if calculate_overlapping:
            no_overlapping = calculate_intersections(paths, path_attributes=path_attributes, ignore_color="white")
            stats["overlapping curves"].append(no_overlapping)
        stats["paths"].append(no_paths)
        stats["curves"].append(no_curves)
        stats["average path length"].append(avg_path_len)
        stats["average curve length"].append(avg_curve_len)
        stats["average curves per path"].append(avg_curve_per_path)
    with open(cache_filename, "w") as cache_file:
        dump(stats, cache_file)
    return stats    

def summarize_statistics(statistics, calculate_overlapping=False):
    count_summary = {
        "images": len(statistics["paths"]),
    }
    dist_summary = {
        ("paths", "median"): np.median(statistics["paths"]),
        ("paths", "\\acrshort{iqr}"): iqr(statistics["paths"]),
        ("curves", "median"): np.median(statistics["curves"]),
        ("curves", "\\acrshort{iqr}"): iqr(statistics["curves"]),
        ("curves / path", "median"): np.median(statistics["average curves per path"]),
        ("curves / path", "\\acrshort{iqr}"): iqr(statistics["average curves per path"]),
        ("path length", "median"): np.median(statistics["average path length"]),
        ("path length", "\\acrshort{iqr}"): iqr(statistics["average path length"]),
        ("curve length", "median"): np.median(statistics["average curve length"]),
        ("curve length", "\\acrshort{iqr}"): iqr(statistics["average curve length"]),
    }
    if calculate_overlapping:
        count_summary.update({
            "images with overlapping curves": np.sum(np.array(statistics["overlapping curves"]) > 0),
        })
        dist_summary.update({
            ("overlap curves", "median"): np.median(statistics["overlapping curves"]),
            ("overlap curves", "\\acrshort{iqr}"): np.median(statistics["overlapping curves"]),
        })
    return count_summary, dist_summary


def convert_integers_to_int(value):
    if type(value) is float and value.is_integer():
        return int(value)
    else:
        return value


# summaries are split into two tables in order to fit page width
count_summaries = {}
dist_summaries = {}
for dataset in datasets:
    is_tonari = dataset == "tonari"
    statistics = get_statistics(data_dir, dataset, calculate_overlapping=is_tonari)
    count_summary, dist_summary = summarize_statistics(statistics, calculate_overlapping=is_tonari)
    count_summaries[dataset] = count_summary
    dist_summaries[dataset] = dist_summary

count_df = pd.DataFrame.from_dict(count_summaries, orient="index", dtype=pd.Int64Dtype())
with open(join(tables, "dataset_count_summaries.tex"), "w") as tex_file:
    count_df.to_latex(buf=tex_file, float_format="%.2f", na_rep="-")
columns = pd.MultiIndex.from_tuples(dist_summaries[list(dist_summaries.keys())[0]].keys())
dist_df = pd.DataFrame.from_dict(dist_summaries, orient="index", columns=columns)
dist_df = dist_df.applymap(convert_integers_to_int)
with open(join(tables, "dataset_dist_summaries.tex"), "w") as tex_file:
    dist_df.T.to_latex(buf=tex_file, float_format="%.2f", multicolumn_format="c", na_rep="-")


paths, _, _ = svg2paths2(join(data_dir, "/tonari/49.svg"))
path_len = [path.length() for path in paths]
plt.hist(path_len)
plt.xlabel("Path length")
plt.ylabel("Frequency")
plt.savefig(join(figures, "path_len.pdf"))
plt.show()
curves = list(chain(*((curve for curve in path) for path in paths)))
curve_len = [curve.length() for curve in curves]
plt.hist(curve_len)
plt.xlabel("Curve length")
plt.ylabel("Frequency")
plt.savefig(join(figures, "curve_len.pdf"))
plt.show()
no_curve_per_path = [len(path) for path in paths]
counts = np.bincount(no_curve_per_path)
plt.bar(range(len(counts)), counts, width=1, align='center')
plt.xticks(range(len(counts)))
plt.xlim([0, len(counts)])
plt.xlabel("Number of curves per path")
plt.ylabel("Frequency")
plt.savefig(join(figures, "no_curve_per_path.pdf"))
plt.show()
