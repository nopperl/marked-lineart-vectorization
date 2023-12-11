from os.path import join

from cairosvg import svg2pdf
import numpy as np
from svgpathtools import CubicBezier, wsvg

report = "report"
figures = join(report, "figures")

a = CubicBezier(4+5j, 3+2j, 1+4j, 3+1j)
b = CubicBezier(4+5j, 3+2j, 1+4j, 3+4j)

out_filename=join(figures, "curve_a.svg")
wsvg(a, filename=out_filename)
svg2pdf(url=out_filename, write_to=out_filename.replace(".svg", ".pdf"))

out_filename=join(figures, "curve_b.svg")
wsvg(b, filename=out_filename)
svg2pdf(url=out_filename, write_to=out_filename.replace(".svg", ".pdf"))

aes = np.absolute(np.array(a.bpoints()).view(float) - np.array(b.bpoints()).view(float))

print(np.mean(aes))

print(np.sum(aes))