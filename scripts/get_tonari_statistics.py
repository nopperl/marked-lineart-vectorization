#!/usr/bin/env python
# coding: utf-8

from os import makedirs
from os.path import join
from collections import Counter, OrderedDict
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
from svgelements import Color
from svgpathtools import svg2paths


input_dir = "../data/clean/tonari"
report = "../report"
figures = join(report, "figures")
tables = join(report, "tables")

makedirs(report, exist_ok=True)
makedirs(figures, exist_ok=True)
makedirs(tables, exist_ok=True)


stroke_colors = []
stroke_widths = []
svg_files = Path(input_dir).rglob("*.svg")
for svg_file in svg_files:
    _, path_attributes = svg2paths(str(svg_file))
    stroke_colors.extend([Color(p["stroke"]).hex for p in path_attributes])
    stroke_widths.extend([float(p["stroke-width"]) for p in path_attributes])


stroke_color_counter = Counter(stroke_colors)
stroke_width_counter = Counter(stroke_widths)
stroke_width_counter = OrderedDict({key: stroke_width_counter[key] for key in sorted(stroke_width_counter.keys())})
stroke_width_counter.pop(2)


df_color = pd.DataFrame.from_dict(stroke_color_counter, orient="index", columns=["Number of curves"])
df_color["color"] = df_color.index
df_color.style.applymap(lambda c: "background-color: " + c, subset=["color"])


schema = ["#000000", "#302010", "#4c3219", "#654321", "#ff00ff", "#b100ff", "#7d00ff", "#0000ff", "#ffff00", "#ff9900", "#ff6700", "#ff0000", "#00ff00", "#00ffbf", "#02fff2", "#00d0ff", "#797878", "#b7b7b7", "#cccccc", "#d9d9d9"]
df_color = pd.DataFrame.from_dict(stroke_color_counter, orient="index", columns=["Number of paths"])
df_color.sort_values(by="Number of paths", inplace=True, ascending=False)
df_color["Color"] = df_color.index
df_color["Part of schema"] = df_color.index.map(lambda x: x in schema)
df_color["hex string"] = df_color.index.map(lambda x: "\\" + x)
df_color.Color = df_color.Color.apply(lambda x: "\\cellcolor{%s}\\color{%s}%s" % (x[1:], x[1:], x[1:]))
df_color = df_color[["hex string", "Color", "Part of schema", "Number of paths"]]
with open(join(tables, "stroke_color_counter.tex"), "w") as tex_file:
    latex = "\n".join(["\\definecolor{%s}{HTML}{%s}" % (x[2:], x[2:]) for x in df_color["hex string"]])
    latex += "\n" + df_color.to_latex(index=False)
    tex_file.write(latex)


f = plt.figure()
f.set_figheight(7)
plt.barh(range(len(stroke_width_counter.keys())), stroke_width_counter.values())
plt.ylim([1, len(stroke_width_counter) - 1])
plt.yticks(range(len(stroke_width_counter.keys())), [format(k, ".2f") for k in stroke_width_counter.keys()])
plt.ylabel("Stroke width")
plt.xlabel("Number of paths")
f.tight_layout()
plt.savefig(join(figures, "stroke_width_counter.pdf"))
