"""Wave 1-B analysis. Reads runs/B/*/curve.jsonl and prints Markdown tables.

Main metric: Crop36 zero-shot top-1 (zs_top1), COCO rsum (sum of R@1/5/10 both ways, max 6).
Per run : final (last eval), E90, AUC (mean over evaluated epochs >= 1), unconverged flag
          (last 3 evals rising, total +0.01 or more on the 0-1 scale; +0.06 for rsum), final train
          loss / ln B, final logit scale and bias, collapse flags (train loss >= 0.98 ln B at the end for
          clip; image or text embedding std < 0.01).
          Also: median of the last 5 evals, dips (eval-to-eval drops >= 0.1 on the 0-1 scale, 0.6 for rsum),
          BN-recalibrated final (from --eval-recal curves, else from recal.json post-hoc) and its range/dips.
Per group (data, loss, batch, epochs, tag): mean over seeds and seed range.
"""
import argparse, glob, json, math, os
from collections import defaultdict


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.expanduser("~/MM_Wave1/runs/B"))
    a = p.parse_args()
    groups, per_run = defaultdict(list), {}
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "curve.jsonl"))):
        d = os.path.dirname(f)
        meta = json.load(open(os.path.join(d, "meta.json")))
        ar = meta["args"]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        cur = [r for r in rows if r["epoch"] >= 1]
        if not cur:
            continue
        key = "zs_top1" if ar["data"] == "crop36" else "rsum"
        tol = 0.01 if key == "zs_top1" else 0.06
        vals = [r[key] for r in cur]
        fin = vals[-1]
        e90 = next((r["epoch"] for r in cur if r[key] >= 0.9 * fin), None) if fin > 0 else None
        tail = vals[-3:]
        last = cur[-1]
        lnb = math.log(ar["batch"])
        fj = os.path.join(d, "final.json")
        status = json.load(open(fj)).get("status") if os.path.exists(fj) else "running"
        flags = []
        if len(tail) == 3 and tail[0] < tail[1] < tail[2] and tail[2] - tail[0] >= tol:
            flags.append("unconverged")
        if ar["loss"] == "clip" and last["train_loss"] >= 0.98 * lnb:
            flags.append("loss~lnB")
        if min(last.get("featstd_img", 1), last.get("featstd_txt", 1)) < 0.01:
            flags.append("emb-collapse")
        if status != "ok":
            flags.append(status)
        med5 = sorted(vals[-5:])[len(vals[-5:]) // 2]
        dips = sum(1 for i in range(1, len(vals)) if vals[i - 1] - vals[i] >= 0.1 * (1 if key == "zs_top1" else 6))
        rk = "recal_" + key
        rvals = [r[rk] for r in cur if rk in r]
        rdips = sum(1 for i in range(1, len(rvals)) if rvals[i - 1] - rvals[i] >= 0.1 * (1 if key == "zs_top1" else 6)) if rvals else None
        rj = os.path.join(d, "recal.json")
        post = json.load(open(rj))["after"][key] if os.path.exists(rj) else None
        m = {"med5": med5, "dips": dips, "recal_final": rvals[-1] if rvals else None, "recal_dips": rdips,
             "posthoc_recal": post,
             "data": ar["data"], "loss": ar["loss"], "batch": ar["batch"], "epochs": ar["epochs"],
             "tag": ar.get("tag", ""), "seed": ar["seed"], "metric": key, "final": fin, "e90": e90,
             "auc": sum(vals) / len(vals), "train_loss_over_lnB": last["train_loss"] / lnb,
             "probe": last.get("probe"), "logit_scale": last.get("logit_scale"),
             "logit_bias": last.get("logit_bias"), "flags": flags, "status": status}
        per_run[meta["name"]] = m
        groups[(ar["data"], ar["loss"], ar["batch"], ar["epochs"], ar.get("tag", ""))].append(m)

    def f(x, nd=4):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    summary = []
    for data in ("crop36", "coco"):
        ks = sorted([k for k in groups if k[0] == data], key=lambda k: (k[2] != 256, k[3], k[4], ["vse0", "vsepp", "clip", "siglip"].index(k[1])))
        if not ks:
            continue
        metric = "zero-shot top-1" if data == "crop36" else "rsum (max 6)"
        print(f"\n### {data} ({metric})\n")
        print("| loss | batch | epochs | tag | seeds | final (mean) | seed range | last-5 median | dips | BN-recal final | recal range | recal dips | E90 | AUC | probe | loss/lnB | scale | bias | flags |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in ks:
            ms = groups[k]
            fins = [x["final"] for x in ms]
            row = {"data": k[0], "loss": k[1], "batch": k[2], "epochs": k[3], "tag": k[4],
                   "seeds": sorted(x["seed"] for x in ms), "final_mean": mean(fins),
                   "seed_range": max(fins) - min(fins), "e90_mean": mean([x["e90"] for x in ms]),
                   "auc_mean": mean([x["auc"] for x in ms]), "probe": mean([x["probe"] for x in ms]),
                   "loss_over_lnB": mean([x["train_loss_over_lnB"] for x in ms]),
                   "logit_scale": mean([x["logit_scale"] for x in ms]),
                   "logit_bias": mean([x["logit_bias"] for x in ms]),
                   "flags": sorted({fl for x in ms for fl in x["flags"]}),
                   "med5": mean([x["med5"] for x in ms]), "dips": sum(x["dips"] for x in ms)}
            rf = [x["recal_final"] if x["recal_final"] is not None else x["posthoc_recal"] for x in ms]
            rf = [v for v in rf if v is not None]
            row["recal_final"] = mean(rf) if len(rf) == len(ms) else None
            row["recal_range"] = (max(rf) - min(rf)) if len(rf) == len(ms) else None
            rd = [x["recal_dips"] for x in ms if x["recal_dips"] is not None]
            row["recal_dips"] = sum(rd) if rd else None
            summary.append(row)
            print(f"| {k[1]} | {k[2]} | {k[3]} | {k[4] or '-'} | {len(ms)} | {f(row['final_mean'])} | "
                  f"{f(row['seed_range'])} | {f(row['med5'])} | {row['dips']} | {f(row['recal_final'])} | "
                  f"{f(row['recal_range'])} | {f(row['recal_dips'])} | {f(row['e90_mean'], 1)} | {f(row['auc_mean'])} | {f(row['probe'])} | "
                  f"{f(row['loss_over_lnB'], 3)} | {f(row['logit_scale'], 2)} | {f(row['logit_bias'], 2)} | "
                  f"{' '.join(row['flags']) or '-'} |")
    json.dump({"groups": summary, "runs": per_run}, open(os.path.join(a.runs, "summary.json"), "w"), indent=1)
    print(f"\n{len(per_run)} runs, summary.json written to {a.runs}")


if __name__ == "__main__":
    main()
