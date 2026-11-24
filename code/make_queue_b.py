"""Build the Wave 1-B queue from smoke timings (appends to jobs.txt by default).

Order: Crop36 30 ep -> Crop36 100 ep -> COCO 15 ep -> COCO 50 ep -> batch-64 auxiliary (Crop36 30 ep),
each with seeds 777, 778, so a stop at the deadline leaves complete earlier sets.
Estimate per job = epochs x (train sec + eval sec / eval_every) x margin, from runs/B_smoke.
"""
import argparse, json, os

LOSSES = ["vse0", "vsepp", "clip", "siglip"]
TIMING_AS = {"vse0": "vsepp"}  # same compute; vse0 was not in the first smoke


def per_epoch(smoke, data, loss, batch, epochs):
    f = os.path.join(smoke, f"{data}_{TIMING_AS.get(loss, loss)}_b{batch}_e1_s777", "curve.jsonl")
    r = [json.loads(l) for l in open(f) if l.strip()][-1]
    ev = max(1, epochs // 30)
    return r["epoch_sec"] + r["eval_sec"] / ev, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", default=os.path.expanduser("~/MM_Wave1/runs/B_smoke"))
    p.add_argument("--out", default=os.path.expanduser("~/MM_Wave1/queue/jobs.txt"))
    p.add_argument("--seeds", default="777,778")
    p.add_argument("--margin", type=float, default=1.2)
    p.add_argument("--print-only", action="store_true")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    plan = [("crop36", 256, 30, LOSSES), ("crop36", 256, 100, LOSSES), ("coco", 256, 15, LOSSES),
            ("coco", 256, 50, LOSSES), ("crop36", 64, 30, ["clip", "siglip"])]
    lines, total, blocks = [], 0.0, []
    for data, batch, ep, losses in plan:
        sub = 0.0
        for seed in seeds:
            for l in losses:
                pe, r = per_epoch(a.smoke, data, l, batch, ep)
                est = max(1, round(pe * ep * a.margin / 60))
                sub += est
                name = f"B_{data}_{l}_b{batch}_e{ep}_s{seed}"
                cmd = (f"cd ~/MM_Wave1/code && python mm_b.py --data {data} --loss {l} --batch {batch} "
                       f"--epochs {ep} --seed {seed}")
                lines.append(f"{name} | {est} | {cmd}")
        blocks.append(f"{data} b{batch} e{ep}: {len(seeds) * len(losses)} jobs, {sub / 60:.1f} h")
        total += sub
    for b in blocks:
        print(b)
    print(f"{len(lines)} jobs, estimated {total / 60:.1f} h")
    if not a.print_only:
        with open(a.out, "a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"appended to {a.out}")


if __name__ == "__main__":
    main()
