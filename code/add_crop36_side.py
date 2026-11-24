"""Recover the original box side (max(w, h) in px) of every Crop36 crop.

Walks FashionLetters36 in the same order as SSL prep_ssl_data.py (ImageSets/Main/{trainval,test}.txt,
objects in XML order), regenerates every crop with the same square_crop rule and checks it is
pixel-identical to crop36_{train,test}.npz['x'] (and the label to ['y']). Only if every crop matches
does it write crop36_{split}_side.npy (float32, same order). The npz files are not modified.
"""
import argparse, hashlib, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

CLASSES = ("tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot",
           *[chr(ord("A") + i) for i in range(26)])
CROP, MARGIN = 96, 1.15


def square_crop(img, x0, y0, x1, y1):  # identical to prep_ssl_data.py
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(round(MARGIN * max(x1 - x0, y1 - y0)))
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = side
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    patch = padded[b + pad:b + pad + side, a + pad:a + pad + side]
    interp = cv2.INTER_AREA if side > CROP else cv2.INTER_LINEAR
    return cv2.resize(patch, (CROP, CROP), interpolation=interp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.expanduser("~/SSL_Wave1"))
    a = p.parse_args()
    fl36 = Path(a.root) / "datasets" / "FashionLetters36"
    out = Path(a.root) / "datasets" / "ssl"
    cls_id = {c.lower(): i for i, c in enumerate(CLASSES)}
    ok = True
    for split, name in (("trainval", "train"), ("test", "test")):
        z = np.load(out / f"crop36_{name}.npz")
        X, Y = z["x"], z["y"]
        ids = [x for x in (fl36 / "ImageSets" / "Main" / f"{split}.txt").read_text().split() if x]
        sides, k, mism_px, mism_y = [], 0, 0, 0
        for i in ids:
            img = cv2.imread(str(fl36 / "JPEGImages" / f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)
            for obj in ET.parse(fl36 / "Annotations" / f"{i}.xml").getroot().iter("object"):
                nm = obj.find("name").text.strip().lower()
                bb = obj.find("bndbox")
                x0, y0, x1, y1 = (float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
                x0, y0 = x0 - 1, y0 - 1
                if k < len(X):
                    mism_px += int(not np.array_equal(square_crop(img, x0, y0, x1, y1), X[k]))
                    mism_y += int(cls_id[nm] != Y[k])
                sides.append(max(x1 - x0, y1 - y0))
                k += 1
        s = np.array(sides, np.float32)
        good = k == len(X) and mism_px == 0 and mism_y == 0
        b = np.digitize(s, [64, 112])
        print(f"{name}: crops {k} (npz {len(X)}), pixel mismatches {mism_px}, label mismatches {mism_y}, "
              f"side min {s.min():.0f} max {s.max():.0f}, buckets small/medium/large {np.bincount(b, minlength=3).tolist()}"
              f" -> {'OK' if good else 'FAIL'}")
        if good:
            path = out / f"crop36_{name}_side.npy"
            np.save(path, s)
            print(f"  wrote {path} md5 {hashlib.md5(open(path, 'rb').read()).hexdigest()}")
        ok &= good
    print("CROP36 SIDE:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
