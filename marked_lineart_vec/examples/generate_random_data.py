#!/usr/bin/env python
from marked_lineart_vec.datasets import generate_visible_paths
from marked_lineart_vec.util import save_as_svg

curves_batch = generate_visible_paths(1, 5, 4, 512, 512, .512)
save_as_svg(curves_batch[0] * 512, "report/figures/synth.svg", 512, 512, stroke_width=0.512)
