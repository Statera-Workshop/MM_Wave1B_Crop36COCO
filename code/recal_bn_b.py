"""Re-evaluate saved Wave 1-B models with recalibrated BatchNorm statistics.

For each run directory with last.pt: rebuild the model from meta.json, evaluate as saved
("before", must equal final.json), then reset every BatchNorm running mean/var, recompute them as a
cumulative average over --n-batches training batches in train mode (no gradient), and evaluate again
("after"). Results go to <run>/recal.json and a Markdown table. Weights are not changed on disk.
Purpose: test whether the late zero-shot dips are BN running-statistic lag (eval-mode artefact).
"""
import argparse, glob, json, os, sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_b as M


def setup(data, dev, crop36_dir, coco_path, cache):
    if data in cache:
        return cache[data]
    if data == "crop36":
        tok = M.Tok(32)
        d = M.load_crop36(crop36_dir, dev)
        table, _, _ = M.crop36_caption_table(tok)
        zs_ids = table.to(dev)[:, 0, [M.TEMPLATES.index(t) for t in M.ZS_TEMPLATES]]
        norm = (torch.tensor([0.5], device=dev).view(1, 1, 1, 1),) * 2
        cache[data] = (tok, d, {"zs_ids": zs_ids}, norm, 1, d["train"]["img"])
    else:
        tok = M.Tok(64)
        d = M.load_coco(coco_path, dev)
        va_ids, _ = tok([c for caps in d["val"]["caps"] for c in caps])
        va_ids = va_ids.view(len(d["val"]["caps"]), -1, 64).to(dev)
        norm = (torch.tensor([0.481, 0.458, 0.408], device=dev).view(1, 3, 1, 1),
                torch.tensor([0.269, 0.261, 0.276], device=dev).view(1, 3, 1, 1))
        cache[data] = (tok, d, {"va_ids": va_ids}, norm, 3, d["train"]["img"])
    return cache[data]


def evaluate(model, data, d, extra, norm, probe_n):
    if data == "crop36":
        return M.evaluate_crop36(model, d, extra["zs_ids"], norm, probe_n)
    return M.evaluate_coco(model, d, extra["va_ids"], norm)


@torch.no_grad()
def recalibrate(model, imgs, norm, n_batches, bs, seed):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over the recalibration batches
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(imgs.shape[0], generator=g)[: n_batches * bs].to(imgs.device)
    model.train()
    for s in range(0, idx.numel(), bs):
        model.encode_image(M.to_float(imgs[idx[s:s + bs]], *norm))
    model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.expanduser("~/MM_Wave1/runs/B/*"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-batches", type=int, default=50)
    p.add_argument("--crop36-dir", default=os.path.expanduser("~/SSL_Wave1/datasets/ssl"))
    p.add_argument("--coco", default=os.path.expanduser("~/MM_Wave1/data/coco_sub/coco_sub.npz"))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    dev = torch.device(a.device)
    cache, rows = {}, []
    print("| run | metric | final.json | before | after | change |")
    print("|---|---|---|---|---|---|")
    for rd in sorted(glob.glob(a.runs)):
        if not os.path.exists(os.path.join(rd, "last.pt")):
            continue
        out = os.path.join(rd, "recal.json")
        if os.path.exists(out) and not a.force:
            r = json.load(open(out))
        else:
            meta = json.load(open(os.path.join(rd, "meta.json")))
            ar = meta["args"]
            tok, d, extra, norm, in_ch, tr_img = setup(ar["data"], dev, a.crop36_dir, a.coco, cache)
            model = M.DualEncoder(ar["loss"], tok.vocab, meta["ctx"], in_ch).to(dev)
            model.load_state_dict(torch.load(os.path.join(rd, "last.pt"), map_location=dev, weights_only=True))
            model.eval()
            key = "zs_top1" if ar["data"] == "crop36" else "rsum"
            fin = json.load(open(os.path.join(rd, "final.json"))).get(key)
            before = evaluate(model, ar["data"], d, extra, norm, ar.get("probe_n", 10000))
            recalibrate(model, tr_img, norm, a.n_batches, ar["batch"], ar["seed"])
            after = evaluate(model, ar["data"], d, extra, norm, ar.get("probe_n", 10000))
            r = {"name": meta["name"], "metric": key, "final_json": fin, "before": before, "after": after,
                 "n_batches": a.n_batches}
            json.dump(r, open(out, "w"), indent=1)
        k = r["metric"]
        b, af = r["before"][k], r["after"][k]
        rows.append(r)
        fj = "-" if r["final_json"] is None else f"{r['final_json']:.4f}"
        print(f"| {r['name']} | {k} | {fj} | {b:.4f} | {af:.4f} | {af - b:+.4f} |", flush=True)
    print(f"\n{len(rows)} runs re-evaluated")


if __name__ == "__main__":
    main()
