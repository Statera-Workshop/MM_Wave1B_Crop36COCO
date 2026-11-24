"""Build the COCO Captions subset for Wave 1-B.

- captions from annotations_trainval2017.zip (CC BY 4.0), read from the zip without extracting
- train: 20,000 images drawn with seed 777 from train2017 (sorted ids); val: 1,000 from val2017
- each image: shorter side -> 128 (bicubic), centre crop 128x128, RGB uint8
- each image keeps its first 5 captions (by annotation id), whitespace-normalised
Images are fetched one by one from images.cocodataset.org and cached as resized .npy files,
so an interrupted run resumes. Output: coco_sub.npz + coco_sub_list.json (ids, md5 of the list).
Images are not redistributed; the list and this script are what get published.
"""
import argparse, hashlib, io, json, os, sys, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{file}"


def fetch(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                raise
            print(f"  retry {t + 1} {url}: {str(e)[:60]}", flush=True)
            time.sleep(2 * (t + 1))


def resize_crop(b, side=128):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    s = side / min(w, h)
    im = im.resize((max(side, round(w * s)), max(side, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - side) // 2, (h - side) // 2
    return np.asarray(im.crop((l, t, l + side, t + side)), dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/MM_Wave1/data/coco_sub"))
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--side", type=int, default=128)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "cache"), exist_ok=True)
    zpath = os.path.join(a.out, "annotations_trainval2017.zip")
    if not os.path.exists(zpath):
        print("downloading annotations (~241 MB) ...", flush=True)
        urllib.request.urlretrieve(ANN_URL, zpath + ".part")
        os.rename(zpath + ".part", zpath)
    z = zipfile.ZipFile(zpath)
    rng = np.random.default_rng(a.seed)
    result, listing = {}, {}
    for split, n in (("train2017", a.n_train), ("val2017", a.n_val)):
        ann = json.loads(z.read(f"annotations/captions_{split}.json"))
        files = {im["id"]: im["file_name"] for im in ann["images"]}
        caps = {}
        for c in sorted(ann["annotations"], key=lambda c: c["id"]):
            caps.setdefault(c["image_id"], []).append(" ".join(c["caption"].strip().split()))
        ids = sorted(i for i in files if len(caps.get(i, [])) >= 5)
        pick = sorted(rng.choice(ids, size=n, replace=False).tolist())
        todo = [i for i in pick if not os.path.exists(os.path.join(a.out, "cache", f"{i}.npy"))]
        print(f"{split}: {len(ids)} eligible, picked {n}, to download {len(todo)}", flush=True)

        def work(i):
            arr = resize_crop(fetch(IMG_URL.format(split=split, file=files[i])), a.side)
            np.save(os.path.join(a.out, "cache", f"{i}.npy"), arr)
            return i

        done, fails, t0 = 0, [], time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(work, i): i for i in todo}
            for f in futs:
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    fails.append((futs[f], str(e)[:80]))
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{len(todo)} {time.time() - t0:.0f}s fails {len(fails)}", flush=True)
        if fails:
            print(f"FAIL {split}: {len(fails)} images not fetched, e.g. {fails[:3]}; run again to retry")
            sys.exit(1)
        key = "train" if split == "train2017" else "val"
        result[f"{key}_img"] = np.stack([np.load(os.path.join(a.out, "cache", f"{i}.npy")) for i in pick])
        result[f"{key}_caps"] = np.array([caps[i][:5] for i in pick])
        result[f"{key}_ids"] = np.array(pick, dtype=np.int64)
        listing[split] = pick
    np.savez(os.path.join(a.out, "coco_sub.npz"), **result)
    lst = json.dumps(listing, sort_keys=True)
    md5 = hashlib.md5(lst.encode()).hexdigest()
    json.dump({"seed": a.seed, "side": a.side, "ids": listing, "md5_of_ids": md5},
              open(os.path.join(a.out, "coco_sub_list.json"), "w"))
    print(f"train {result['train_img'].shape} val {result['val_img'].shape} ids md5 {md5}")
    print("PREP COCO: OK")


if __name__ == "__main__":
    main()
