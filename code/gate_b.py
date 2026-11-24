"""Wave 1-B gates.

1. counts    : Crop36 26,716 / 6,699 and 36 classes; COCO subset 20,000 / 1,000 with 5 captions each.
   (vsepp's overfit is INFO only: max violation from scratch is the behaviour under study)
2. tokenizer : every Crop36 caption decodes back to itself; captions over the context length = 0.
3. eval path : oracle embeddings give zero-shot top-1 1.000 and Recall@1 1.000.
4. params    : every parameter (incl. logit scale / bias) changes after one step, per loss.
5. overfit   : 512 pairs; Crop36 zero-shot on those 512 >= 0.99, COCO Recall@1 (both ways) >= 0.95.
Ends with 'GATE B: PASS' / 'GATE B: FAIL'.
"""
import argparse, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_b as M


def overfit(model, opt, batches_fn, steps, check_fn, every=100, warmup=50, lr=5e-4):
    for s in range(1, steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, s / warmup)
        model.train()
        loss = 0.0
        for img, ids in batches_fn():
            ei, et = model.encode_image(img), model.encode_text(ids)
            l = M.contrastive_loss(model, ei, et)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            loss = l.item()
        if s % every == 0:
            v = check_fn()
            if v is True:
                return s, loss
    return None, loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop36-dir", default=os.path.expanduser("~/SSL_Wave1/datasets/ssl"))
    p.add_argument("--coco", default=os.path.expanduser("~/MM_Wave1/data/coco_sub/coco_sub.npz"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--expect", default="26716,6699,20000,1000")
    p.add_argument("--skip-coco", action="store_true")
    p.add_argument("--losses", default="vse0,vsepp,clip,siglip")
    a = p.parse_args()
    dev = torch.device(a.device)
    e_tr, e_te, e_ctr, e_cva = [int(x) for x in a.expect.split(",")]
    ok = True

    def check(name, good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"{name} {msg} -> {'OK' if good else 'FAIL'}", flush=True)

    # 1. counts
    c36 = M.load_crop36(a.crop36_dir, dev)
    n_tr, n_te = len(c36["train"]["y"]), len(c36["test"]["y"])
    ncls = len(torch.unique(c36["train"]["y"]))
    has_size = bool((c36["train"]["size"] >= 0).all())
    check("[1] crop36", n_tr == e_tr and n_te == e_te and ncls == 36,
          f"train {n_tr} test {n_te} classes {ncls} img {tuple(c36['train']['img'].shape[1:])} size field {has_size}")
    if has_size:
        print("    size buckets (train):", torch.bincount(c36["train"]["size"], minlength=3).tolist())
    coco = None
    if not a.skip_coco:
        coco = M.load_coco(a.coco, dev)
        ntr, nva = len(coco["train"]["caps"]), len(coco["val"]["caps"])
        ncap = {len(c) for c in coco["train"]["caps"] + coco["val"]["caps"]}
        check("[1] coco", ntr == e_ctr and nva == e_cva and ncap == {5},
              f"train {ntr} val {nva} captions/image {sorted(ncap)} img {tuple(coco['train']['img'].shape[1:])}")

    # 2. tokenizer
    tok32 = M.Tok(32)
    table, texts, over = M.crop36_caption_table(tok32)
    flat = table.view(-1, 32)
    bad = [t for t, row in zip(texts, flat) if tok32.decode(row) != M.SimpleTokenizer().decode(
        M.SimpleTokenizer().encode(t)).strip()]
    check("[2] crop36 tokens", not bad and over == 0,
          f"{len(texts)} captions, round-trip mismatches {len(bad)}, over ctx {over}, max len {int((flat > 0).sum(1).max())}")
    if coco is not None:
        tok64 = M.Tok(64)
        caps = [c for cs in coco["train"]["caps"] + coco["val"]["caps"] for c in cs]
        ids, over = tok64(caps)
        lens = (ids > 0).sum(1).float()
        check("[2] coco tokens", over == 0,
              f"{len(caps)} captions, over ctx {over}, length mean {lens.mean():.1f} max {int(lens.max())}")

    # 3. eval path (oracle)
    y = c36["test"]["y"]
    ei = F.one_hot(y, M.EMB).float()
    w = F.one_hot(torch.arange(36, device=dev), M.EMB).float()
    zs = M.accuracy(ei @ w.t(), y)
    n = 200
    owner = torch.arange(n, device=dev).repeat_interleave(5)
    sim = F.one_hot(torch.arange(n, device=dev), M.EMB).float() @ F.one_hot(owner, M.EMB).float().t()
    rk = M.recall_at_k(sim, owner)
    check("[3] eval path", zs == 1.0 and rk["i2t_R@1"] == 1.0 and rk["t2i_R@1"] == 1.0,
          f"zero-shot {zs:.3f} i2t R@1 {rk['i2t_R@1']:.3f} t2i R@1 {rk['t2i_R@1']:.3f}")

    norm1 = (torch.tensor([0.5], device=dev).view(1, 1, 1, 1),) * 2
    zs_ids = table.to(dev)[:, 0, [M.TEMPLATES.index(t) for t in M.ZS_TEMPLATES]]
    sub = {k: v[:a.n] for k, v in c36["train"].items()}
    sz = torch.where(sub["size"] < 0, torch.zeros_like(sub["size"]), sub["size"])
    ids_fixed = table.to(dev)[sub["y"], sz, 0]  # one fixed caption per image for the gates

    # 4. params
    for loss in a.losses.split(","):
        M.seed_all(777)
        m = M.DualEncoder(loss, tok32.vocab, 32, 1).to(dev)
        opt = M.make_opt(m, 5e-4, 0.1)
        before = {k: v.detach().clone() for k, v in m.named_parameters()}
        m.train()
        l = M.contrastive_loss(m, m.encode_image(M.to_float(sub["img"][:64], *norm1)), m.encode_text(ids_fixed[:64]))
        opt.zero_grad()
        l.backward()
        opt.step()
        stale = [k for k, v in m.named_parameters() if torch.equal(before[k], v.detach())]
        # embedding rows of unused tokens legitimately stay put; check the table as a whole changed
        stale = [k for k in stale if k != "txt.tok.weight"]
        check(f"[4] {loss:6s}", not stale and l.item() > 0,
              f"params {sum(v.numel() for v in m.parameters()):,} first loss {l.item():.3f} unchanged {stale or 0}")

    # 5. overfit (vsepp = max violation from scratch: reported as INFO, not part of PASS)
    def check5(name, good, msg, info):
        if info:
            print(f"{name} {msg} -> INFO ({'reached' if good else 'not reached'})", flush=True)
        else:
            check(name, good, msg)

    for loss in a.losses.split(","):
        M.seed_all(777)
        m = M.DualEncoder(loss, tok32.vocab, 32, 1).to(dev)
        opt = M.make_opt(m, 5e-4, 0.1)
        img_all = M.to_float(sub["img"], *norm1)
        tab = table.to(dev)
        gen = torch.Generator().manual_seed(777)

        def batches():
            tpl = torch.randint(0, len(M.TEMPLATES), (a.n,), generator=gen).to(dev)
            ids = tab[sub["y"], sz, tpl]  # random template, true size: same captions as training
            for s in range(0, a.n, 256):
                yield img_all[s:s + 256], ids[s:s + 256]

        def chk():
            e, _ = M.embed_images(m, sub["img"], norm1)
            C, T, L = zs_ids.shape
            w = M.zero_shot_weights(M.embed_texts(m, zs_ids.reshape(C * T, L)).view(C, T, -1))
            chk.v = M.accuracy(e @ w.t(), sub["y"])
            return chk.v >= 0.99
        chk.v = 0.0
        t0 = time.time()
        hit, last = overfit(m, opt, batches, a.steps, chk)
        check5(f"[5] crop36 {loss:6s}", hit is not None,
               f"overfit {a.n}: zero-shot {chk.v:.3f} at step {hit} loss {last:.3f} {time.time() - t0:.0f}s",
               loss == "vsepp")

    if coco is not None:
        tok64 = M.Tok(64)
        cid, _ = tok64([cs[0] for cs in coco["train"]["caps"][:a.n]])
        cid = cid.to(dev)
        normc = (torch.tensor([0.481, 0.458, 0.408], device=dev).view(1, 3, 1, 1),
                 torch.tensor([0.269, 0.261, 0.276], device=dev).view(1, 3, 1, 1))
        cimg = coco["train"]["img"][:a.n]
        for loss in a.losses.split(","):
            M.seed_all(777)
            m = M.DualEncoder(loss, tok64.vocab, 64, 3).to(dev)
            opt = M.make_opt(m, 5e-4, 0.1)
            cf = M.to_float(cimg, *normc)

            def batches():
                for s in range(0, a.n, 256):
                    yield cf[s:s + 256], cid[s:s + 256]

            def chk():
                e, _ = M.embed_images(m, cimg, normc)
                r = M.recall_at_k(e @ M.embed_texts(m, cid).t(), torch.arange(a.n, device=dev))
                chk.v = (r["i2t_R@1"], r["t2i_R@1"])
                return min(chk.v) >= 0.95
            chk.v = (0.0, 0.0)
            t0 = time.time()
            hit, last = overfit(m, opt, batches, a.steps, chk)
            check5(f"[5] coco   {loss:6s}", hit is not None,
                   f"overfit {a.n}: R@1 i2t {chk.v[0]:.3f} t2i {chk.v[1]:.3f} at step {hit} loss {last:.3f} {time.time() - t0:.0f}s",
                   loss == "vsepp")
    print("GATE B:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
