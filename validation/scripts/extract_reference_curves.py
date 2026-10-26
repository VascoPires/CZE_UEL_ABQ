"""Extract registered graphical references; never creates or executes FE models.

Usage: python validation/scripts/extract_reference_curves.py --out validation/cases
Refuses to overwrite CSVs. Source PDF hashes, path IDs, axes and uncertainty
are in validation/reference_extraction.json. No origin shifts or curve fitting.
"""
import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from pdf_paths import paths

VALIDATION = Path(__file__).resolve().parents[1]


def centerline(parts, count=80):
    """Midpoint of vertical intersections with a filled stroke outline.

    Only registered single-valued analytical branches use this operation.
    The outline closes explicitly; no extrapolation at stroke endpoints.
    """
    polygon = np.vstack(parts)
    outlines = [np.vstack([p, p[0]]) for p in parts]
    xs = np.linspace(polygon[:, 0].min()+0.6, polygon[:, 0].max()-0.6, count)
    result = []
    for x in xs:
        crossings = []
        for outline in outlines:
            for a, b in zip(outline[:-1], outline[1:]):
                if min(a[0], b[0]) <= x < max(a[0], b[0]):
                    crossings.append(a[1] + (x-a[0])*(b[1]-a[1])/(b[0]-a[0]))
        # The black curve is interrupted by overplotted coloured markers.
        # Retain only visible outline samples; do not fill the missing spans.
        if not crossings:
            continue
        if max(crossings)-min(crossings) > 2:
            raise ValueError("Registered stroke has ambiguous intersections.")
        result.append([x, (min(crossings)+max(crossings))/2])
    if len(result) < 20:
        raise ValueError("Insufficient visible analytical stroke samples.")
    return np.asarray(result)


def extract(spec, cache):
    pdf = VALIDATION / "references" / (spec["key"]+".pdf")
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != spec["pdf_sha256"]:
        raise ValueError("Source PDF changed; re-audit the path IDs: "+str(pdf))
    key = (str(pdf), spec["pdf_page"])
    if key not in cache:
        cache[key] = paths(pdf, spec["pdf_page"])
    objects = cache[key]
    points = []
    if spec["method"] == "filled_stroke_centerline":
        points = centerline(objects[spec["path_ids"][0]]["parts"])
    elif spec["method"] == "circle_centers":
        for index in spec["path_ids"]:
            item = objects[index]
            x, y, xx, yy = item["bbox"]
            if item["op"] != "f*" or not (5.9 < xx-x < 6.1 and 5.9 < yy-y < 6.1):
                raise ValueError("Registered marker geometry changed.")
            points.append([(x+xx)/2, (y+yy)/2])
    elif spec["method"] == "square_centers":
        for part in objects[spec["path_ids"][0]]["parts"]:
            part = np.asarray(part)
            lower, upper = part.min(axis=0), part.max(axis=0)
            if not (len(part) == 5 and np.all((upper-lower > 1.5) & (upper-lower < 1.6))):
                raise ValueError("Registered square geometry changed.")
            points.append((lower+upper)/2)
    else:
        raise ValueError("Unknown extraction method")
    points = np.asarray(points)
    x0, y0, x1, y1 = spec["axis_pdf_points"]
    points[:, 0] = (points[:, 0]-x0)/(x1-x0)*spec["axis_max"][0]
    points[:, 1] = (points[:, 1]-y0)/(y1-y0)*spec["axis_max"][1]
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads((VALIDATION / "reference_extraction.json").read_text())["curves"]
    targets = [args.out/s["case"]/"data"/s["filename"] for s in specs]
    if any(p.exists() for p in targets):
        parser.error("Output exists; use a new directory to preserve prior extractions.")
    cache = {}
    extracted = [extract(s, cache) for s in specs]  # Validate all before writing.
    for spec, target, points in zip(specs, targets, extracted):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="") as handle:
            handle.write("# source: {key}, {figure}, printed p. {printed_page}, PDF page {pdf_page}\n".format(**spec))
            handle.write("# digitised "+date.today().isoformat()+" with pypdf vector paths; "+spec["method"]+"\n")
            handle.write("# graphical estimates, not original raw data; uncertainty_mm_N: "+str(spec["uncertainty_mm_N"])+"\n")
            handle.write("# source_pdf_sha256: "+spec["pdf_sha256"]+"; axes/path IDs: validation/reference_extraction.json\n")
            handle.write("# series: "+spec["series"]+"; retain path order including snap-back; no smoothing/shift\n")
            writer = csv.writer(handle)
            writer.writerow(["displacement_mm", "load_N"])
            writer.writerows([[format(x,".10g"),format(y,".10g")] for x,y in points])
        print(spec["case"], len(points), "points ->", target)


if __name__ == "__main__":
    main()
