#!/usr/bin/env python
from argparse import ArgumentParser
from itertools import chain

from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, wsvg, svg2paths2

def convert_to_cubic(path, recursive=True):
    if type(path) is Line:
        yield CubicBezier(path.start, path.start, path.end, path.end)
    elif type(path) is QuadraticBezier:
        yield CubicBezier(path.start, (path.start + 2 * path.control) / 3, (path.end + 2 * path.control) / 3, path.end)
    elif type(path) is CubicBezier:
        yield path
    elif type(path) is Path:
        if not recursive:
            if len(path._segments) == 1:
                segment = next(iter(path))
                if type(segment) is Line:
                    yield CubicBezier(segment.start, segment.start, segment.end, segment.end)
                elif type(segment) is QuadraticBezier:
                    yield CubicBezier(segment.start, (segment.start + 2 * segment.control) / 3, (segment.end + 2 * segment.control) / 3, segment.end)
                else:
                    yield segment
        else:
            for segment in iter(path):
                if type(segment) is Line:
                    yield CubicBezier(segment.start, segment.start, segment.end, segment.end)
                elif type(segment) is QuadraticBezier:
                    yield CubicBezier(segment.start, (segment.start + 2 * segment.control) / 3, (segment.end + 2 * segment.control) / 3, segment.end)
                else:
                    yield segment
    else:
        yield path


def all_paths_as_cubic(paths, recursive=True):
    return list(chain(*(convert_to_cubic(path, recursive=recursive) for path in paths)))


def main(svg_file):
    paths, _, svg_attributes = svg2paths2(svg_file)
    curves = all_paths_as_cubic(paths)
    path_attributes = [{"stroke": "black", "stroke-linecap": "round", "fill": "none"}] * len(curves)
    wsvg(curves, attributes=path_attributes, svg_attributes=svg_attributes, filename=svg_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("svg_file")
    args = parser.parse_args()
    main(args.svg_file)
