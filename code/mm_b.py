"""PAI-SG Multimodal Wave 1-B: image-text contrastive convergence runs.

Losses : vse0 (sum of hinges, the VSE++ paper's baseline), vsepp (max-of-hinges, margin 0.2), clip (symmetric InfoNCE, learnable temperature,
         init 1/0.07, max 100), siglip (pairwise sigmoid, learnable scale init 10, bias init -10).
Sources: CLIP / SigLIP losses follow open_clip loss.py (MIT); VSE++ follows the paper and
         fartashf/vsepp ContrastiveLoss(max_violation=True); tokenizer = OpenAI CLIP
         simple_tokenizer.py + bpe_simple_vocab_16e6 (MIT), vendored in third_party/.
Data   : crop36 (Crop36-Text, captions from class name + box-size word) or coco (subset npz).

[Unified] all encoders from scratch; image ResNet-18 (BasicBlock, torchvision layout),
          text Transformer 4 layers / width 256 / 4 heads, causal mask, EOT pooling;
          both projected to a 256-d L2-normalised embedding.
[Unified] AdamW lr 5e-4, wd 0.1 (not on norms, biases, embeddings, logit scale/bias),
          linear warm-up 500 steps then constant; batch 256; no augmentation (no flips).
[Unified] in-batch pairs with the same caption text count as negatives (standard CLIP).
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party"))
from simple_tokenizer import SimpleTokenizer  # noqa: E402

EMB = 256

# ----------------------------------------------------------------------------- captions
CLASSES = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag",
           "ankle_boot"] + [chr(ord("A") + i) for i in range(26)]
SIZES = ["small", "medium", "large"]
TEMPLATES = ["a {size} {name}", "a photo of a {size} {name}", "{name}, {size}", "a {name}",
             "a photo of a {name}", "a {size} picture of a {name}", "an image of a {name}",
             "the {name}, {size} size"]
ZS_TEMPLATES = [t for t in TEMPLATES if "{size}" not in t]  # size-free prompts for zero-shot


def class_text(c):
    nice = {"tshirt": "t-shirt", "ankle_boot": "ankle boot"}
    return nice.get(c, c) if len(c) > 1 else f"letter {c}"


def size_bucket(side):
    """Original box side (max(w, h), px) -> 0 small (<64), 1 medium (64-112), 2 large (>=112)."""
    return np.digitize(side, [64, 112]).astype(np.int64)


class Tok:
    def __init__(self, ctx):
        self.t, self.ctx = SimpleTokenizer(), ctx
        self.sot, self.eot = self.t.encoder["<|startoftext|>"], self.t.encoder["<|endoftext|>"]
        self.vocab = len(self.t.encoder)

    def __call__(self, texts):
        out = np.zeros((len(texts), self.ctx), np.int64)
        over = 0
        for i, s in enumerate(texts):
            ids = [self.sot] + self.t.encode(s) + [self.eot]
            if len(ids) > self.ctx:
                over += 1
                ids = ids[:self.ctx - 1] + [self.eot]
            out[i, :len(ids)] = ids
        return torch.from_numpy(out), over

    def decode(self, row):
        ids = [int(x) for x in row if int(x) not in (0, self.sot, self.eot)]
        return self.t.decode(ids).strip()


# ----------------------------------------------------------------------------- data
def load_crop36(path_dir, device):
    """Returns dict(img uint8 (N,1,96,96), y, size) per split. Size bucket from the original box side
    in crop36_{split}_side.npy (made by add_crop36_side.py); -1 when that file is absent."""
    out = {}
    for split in ("train", "test"):
        z = np.load(os.path.join(path_dir, f"crop36_{split}.npz"))
        x, y = z["x"], z["y"].astype(np.int64)
        assert x.dtype == np.uint8 and x.ndim == 3, f"crop36 x expected uint8 (N,96,96), got {x.dtype} {x.shape}"
        sp = os.path.join(path_dir, f"crop36_{split}_side.npy")
        if os.path.exists(sp):
            side = np.load(sp)
            assert len(side) == len(y), "side file length mismatch"
            sb = size_bucket(side)
        else:
            sb = np.full(len(y), -1, np.int64)
        out[split] = {"img": torch.from_numpy(np.ascontiguousarray(x[:, None])).to(device),
                      "y": torch.from_numpy(y).to(device), "size": torch.from_numpy(sb).to(device)}
    return out


def crop36_caption_table(tok):
    """Token table [class, size, template] -> tokens; returns tokens (36,3,T,ctx) and texts."""
    texts = []
    for c in CLASSES:
        for s in SIZES:
            for t in TEMPLATES:
                texts.append(t.format(size=s, name=class_text(c)))
    toks, over = tok(texts)
    return toks.view(len(CLASSES), len(SIZES), len(TEMPLATES), -1), texts, over


def load_coco(path, device):
    z = np.load(path, allow_pickle=False)
    out = {}
    for split in ("train", "val"):
        x = z[f"{split}_img"]  # (N,128,128,3) uint8
        out[split] = {"img": torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device),
                      "caps": [list(c) for c in z[f"{split}_caps"]]}  # (N,5) str
    return out


def to_float(img_u8, ch_mean, ch_std):
    x = img_u8.float().div_(255.0)
    return (x - ch_mean) / ch_std


# ----------------------------------------------------------------------------- models
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + idt)


class ResNet18(nn.Module):
    """torchvision-layout ResNet-18 (conv1/bn1/layer1-4), global average pool -> 512."""

    def __init__(self, in_ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cfg, cin, layers = [(64, 1), (128, 2), (256, 2), (512, 2)], 64, []
        for cout, stride in cfg:
            layers.append(nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout)))
            cin = cout
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        self.out_dim = 512

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)


class TextEncoder(nn.Module):
    """CLIP-style text Transformer: token + positional embedding, causal pre-LN blocks, EOT pooling."""

    def __init__(self, vocab, ctx, width=256, layers=4, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, width)
        self.pos = nn.Parameter(torch.zeros(ctx, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(width)
        self.register_buffer("mask", torch.triu(torch.full((ctx, ctx), float("-inf")), 1), persistent=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos, std=0.01)
        self.out_dim = width

    def forward(self, ids):
        x = self.tok(ids) + self.pos[: ids.shape[1]]
        x = self.tr(x, mask=self.mask[: ids.shape[1], : ids.shape[1]], is_causal=True)
        x = self.ln(x)
        eot = ids.argmax(1)  # EOT has the largest id in the vocabulary (as in CLIP)
        return x[torch.arange(ids.shape[0], device=ids.device), eot]


class DualEncoder(nn.Module):
    def __init__(self, loss, vocab, ctx, in_ch):
        super().__init__()
        self.loss_name = loss
        self.img = ResNet18(in_ch)
        self.txt = TextEncoder(vocab, ctx)
        self.img_proj = nn.Linear(self.img.out_dim, EMB, bias=False)
        self.txt_proj = nn.Linear(self.txt.out_dim, EMB, bias=False)
        if loss == "clip":
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        elif loss == "siglip":
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def encode_image(self, x, feats=False):
        f = self.img(x)
        e = F.normalize(self.img_proj(f), dim=-1)
        return (e, f) if feats else e

    def encode_text(self, ids):
        return F.normalize(self.txt_proj(self.txt(ids)), dim=-1)


def contrastive_loss(model, ei, et):
    n = ei.shape[0]
    if model.loss_name == "clip":
        scale = model.logit_scale.clamp(max=math.log(100)).exp()
        logits = scale * ei @ et.t()
        lab = torch.arange(n, device=ei.device)
        return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab)) / 2
    if model.loss_name == "siglip":
        logits = (model.logit_scale.exp() * ei @ et.t() + model.logit_bias).float()
        lab = 2 * torch.eye(n, device=ei.device) - 1
        return -F.logsigmoid(lab * logits).sum() / n
    if model.loss_name == "vsepp":  # max violation, margin 0.2
        s = ei @ et.t()
        d = s.diag().view(n, 1)
        eye = torch.eye(n, dtype=torch.bool, device=ei.device)
        cost_s = (0.2 + s - d).clamp(min=0).masked_fill(eye, 0)      # image -> wrong captions
        cost_im = (0.2 + s - d.t()).clamp(min=0).masked_fill(eye, 0)  # caption -> wrong images
        return cost_s.max(1)[0].sum() + cost_im.max(0)[0].sum()
    if model.loss_name == "vse0":  # VSE0 of the VSE++ paper: sum of hinges over all negatives
        s = ei @ et.t()
        d = s.diag().view(n, 1)
        eye = torch.eye(n, dtype=torch.bool, device=ei.device)
        cost_s = (0.2 + s - d).clamp(min=0).masked_fill(eye, 0)
        cost_im = (0.2 + s - d.t()).clamp(min=0).masked_fill(eye, 0)
        return cost_s.sum() + cost_im.sum()
    raise ValueError(model.loss_name)


def make_opt(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim < 2 or "logit" in n or n.startswith("txt.tok") or n.startswith("txt.pos") or "bn" in n or "ln" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.98), eps=1e-6)


def load_image_init(img_encoder, path):
    """Load an SSL ResNet-18 backbone (torchvision keys, 3-channel grayscale input x/255) into the image encoder.
    Strips common prefixes, drops fc.*, and for a 1-channel encoder (Crop36 input u = 2x - 1) folds conv1 over the
    three identical input channels and halves it (W.x = W/2.u + const; the constant is removed by the next BatchNorm).
    Every encoder parameter must be loaded; returns (n_loaded, dropped_keys)."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    out, dropped = {}, []
    for k, v in sd.items():
        k2 = k
        for pre in ("module.", "net.", "backbone.", "encoder."):
            if k2.startswith(pre):
                k2 = k2[len(pre):]
        if k2.startswith("fc."):
            dropped.append(k)
            continue
        out[k2] = v
    w = out.get("conv1.weight")
    tgt = img_encoder.conv1.weight
    if w is not None and w.shape[1] == 3 and tgt.shape[1] == 1:
        out["conv1.weight"] = w.sum(1, keepdim=True) / 2
    want = img_encoder.state_dict()
    missing = [k for k in want if k not in out]
    unexpected = [k for k in out if k not in want]
    shape_bad = [k for k in want if k in out and out[k].shape != want[k].shape]
    if missing or unexpected or shape_bad:
        raise RuntimeError(f"init_img {path}: missing {missing[:5]} unexpected {unexpected[:5]} shape {shape_bad[:5]}")
    img_encoder.load_state_dict(out, strict=True)
    return len(out), dropped


# ----------------------------------------------------------------------------- evaluation
def accuracy(logits, y):
    # integer count / n: CUDA mean() multiplies by 1/n and can return 0.99999994 for a perfect score
    return (logits.argmax(1) == y).sum().item() / y.numel()


def linear_probe(ftr, ytr, fte, yte, n_cls, steps=300, lr=0.05):
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    a, b = (ftr - mu) / sd, (fte - mu) / sd
    W = torch.zeros(a.shape[1], n_cls, device=a.device, requires_grad=True)
    c = torch.zeros(n_cls, device=a.device, requires_grad=True)
    opt = torch.optim.Adam([W, c], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(a @ W + c, ytr).backward()
            opt.step()
    with torch.no_grad():
        return accuracy(b @ W + c, yte)


@torch.no_grad()
def embed_images(model, imgs, norm, bs=500):
    model.eval()
    es, fs = [], []
    for s in range(0, imgs.shape[0], bs):
        e, f = model.encode_image(to_float(imgs[s:s + bs], *norm), feats=True)
        es.append(e)
        fs.append(f)
    return torch.cat(es), torch.cat(fs)


@torch.no_grad()
def embed_texts(model, ids, bs=1000):
    model.eval()
    return torch.cat([model.encode_text(ids[s:s + bs]) for s in range(0, ids.shape[0], bs)])


def zero_shot_weights(text_emb_per_class):
    """(C, T, D) prompt embeddings -> (C, D) normalised class weights (prompt ensemble)."""
    return F.normalize(F.normalize(text_emb_per_class, dim=-1).mean(1), dim=-1)


def recall_at_k(sim, img_of_text, ks=(1, 5, 10)):
    """sim (N_img, N_txt). i2t: an image hits if any of its captions is in the top k.
    t2i: a caption hits if its image is in the top k."""
    out = {}
    n_img = sim.shape[0]
    top_t = sim.topk(max(ks), dim=1).indices                    # (N_img, K) text ids
    hit_owner = img_of_text[top_t] == torch.arange(n_img, device=sim.device)[:, None]
    top_i = sim.t().topk(max(ks), dim=1).indices                # (N_txt, K) image ids
    hit_img = top_i == img_of_text[:, None]
    for k in ks:
        out[f"i2t_R@{k}"] = hit_owner[:, :k].any(1).sum().item() / n_img
        out[f"t2i_R@{k}"] = hit_img[:, :k].any(1).sum().item() / hit_img.shape[0]
    return out


def evaluate_crop36(model, data, zs_ids, norm, probe_n):
    te, tr = data["test"], data["train"]
    ei, fi = embed_images(model, te["img"], norm)
    C, T, L = zs_ids.shape
    et = embed_texts(model, zs_ids.view(C * T, L)).view(C, T, -1)
    w = zero_shot_weights(et)
    rec = {"zs_top1": accuracy(ei @ w.t(), te["y"])}
    _, ftr = embed_images(model, tr["img"][:probe_n], norm)
    rec["probe"] = linear_probe(ftr, tr["y"][:probe_n], fi, te["y"], len(CLASSES))
    rec["featstd_img"] = ei.std(0).mean().item()
    rec["featstd_txt"] = et.reshape(C * T, -1).std(0).mean().item()
    return rec


def evaluate_coco(model, data, val_ids, norm):
    va = data["val"]
    ei, _ = embed_images(model, va["img"], norm)
    et = embed_texts(model, val_ids.view(-1, val_ids.shape[-1]))
    n = ei.shape[0]
    owner = torch.arange(n, device=ei.device).repeat_interleave(val_ids.shape[1])
    rec = recall_at_k(ei @ et.t(), owner)
    rec["rsum"] = sum(v for k, v in rec.items())
    rec["featstd_img"] = ei.std(0).mean().item()
    rec["featstd_txt"] = et.std(0).mean().item()
    return rec


@torch.no_grad()
def recalibrated_copy(model, imgs, norm, n_batches, bs, seed):
    """Copy of the model whose BatchNorm running stats are recomputed (cumulative average) over
    n_batches training batches. Uses its own generator, so the training RNG stream is untouched."""
    m2 = copy.deepcopy(model)
    for m in m2.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(imgs.shape[0], generator=g)[: n_batches * bs].to(imgs.device)
    m2.train()
    for s in range(0, idx.numel(), bs):
        m2.encode_image(to_float(imgs[idx[s:s + bs]], *norm))
    m2.eval()
    return m2


# ----------------------------------------------------------------------------- run
def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args):
    name = f"{args.data}_{args.loss}_b{args.batch}_e{args.epochs}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.out, name)
    if os.path.exists(os.path.join(out_dir, "final.json")) and not args.force:
        print(f"SKIP {name}: final.json exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(args.device)
    seed_all(args.seed)
    ctx = 32 if args.data == "crop36" else 64
    tok = Tok(ctx)
    if args.data == "crop36":
        data = load_crop36(args.crop36_dir, dev)
        table, texts, over = crop36_caption_table(tok)
        table = table.to(dev)
        zs_ids = table[:, 0, [TEMPLATES.index(t) for t in ZS_TEMPLATES]]  # size-free prompts
        norm = (torch.tensor([0.5], device=dev).view(1, 1, 1, 1), torch.tensor([0.5], device=dev).view(1, 1, 1, 1))
        in_ch, n_tr = 1, data["train"]["y"].shape[0]
        if (data["train"]["size"] < 0).any():
            print("NOTE: no box-size field; size words are drawn uniformly (see README)")
    else:
        data = load_coco(args.coco, dev)
        tr_ids, over_tr = tok([c for caps in data["train"]["caps"] for c in caps])
        va_ids, over_va = tok([c for caps in data["val"]["caps"] for c in caps])
        over = over_tr + over_va
        tr_ids = tr_ids.view(len(data["train"]["caps"]), -1, ctx).to(dev)
        va_ids = va_ids.view(len(data["val"]["caps"]), -1, ctx).to(dev)
        norm = (torch.tensor([0.481, 0.458, 0.408], device=dev).view(1, 3, 1, 1),
                torch.tensor([0.269, 0.261, 0.276], device=dev).view(1, 3, 1, 1))
        in_ch, n_tr = 3, tr_ids.shape[0]
    seed_all(args.seed)
    model = DualEncoder(args.loss, tok.vocab, ctx, in_ch).to(dev)
    init_info = None
    if args.init_img:
        n_loaded, dropped = load_image_init(model.img, args.init_img)
        init_info = {"path": args.init_img.replace(os.path.expanduser("~"), "~"), "tensors": n_loaded, "dropped": dropped}
        print(f"init_img {args.init_img}: {n_loaded} tensors loaded strictly, dropped {dropped}")
    opt = make_opt(model, args.lr, args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = math.ceil(n_tr / args.batch) if not args.drop_last else n_tr // args.batch
    eval_every = args.eval_every or max(1, args.epochs // 30)
    meta = {"name": name, "args": vars(args), "n_params": n_params, "n_train": n_tr, "ctx": ctx,
            "tokens_over_ctx": over, "steps_per_epoch": steps_per_epoch, "eval_every": eval_every,
            "ln_batch": math.log(args.batch), "torch": torch.__version__, "init_img": init_info,
            "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    cf = open(os.path.join(out_dir, "curve.jsonl"), "w")
    g = torch.Generator().manual_seed(args.seed)
    print(f"START {name} params={n_params} n_train={n_tr} steps/epoch={steps_per_epoch} device={meta['device_name']}", flush=True)

    def log_eval(ep, extra):
        t0 = time.time()
        r = (evaluate_crop36(model, data, zs_ids, norm, args.probe_n) if args.data == "crop36"
             else evaluate_coco(model, data, va_ids, norm))
        rec = {"epoch": ep, **r, **extra, "eval_sec": time.time() - t0}
        if args.eval_recal:
            t1 = time.time()
            imgs = data["train"]["img"]
            m2 = recalibrated_copy(model, imgs, norm, args.recal_batches, args.batch, args.seed + ep)
            r2 = (evaluate_crop36(m2, data, zs_ids, norm, args.probe_n) if args.data == "crop36"
                  else evaluate_coco(m2, data, va_ids, norm))
            rec.update({f"recal_{k}": v for k, v in r2.items()})
            rec["recal_sec"] = time.time() - t1
            del m2
        if args.loss in ("clip", "siglip"):
            rec["logit_scale"] = model.logit_scale.exp().item()
        if args.loss == "siglip":
            rec["logit_bias"] = model.logit_bias.item()
        cf.write(json.dumps(rec) + "\n")
        cf.flush()
        return rec

    key = "zs_top1" if args.data == "crop36" else "rsum"
    rec = log_eval(0, {})
    print(f"ep 0 {key} {rec[key]:.4f}", flush=True)
    step = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        perm = torch.randperm(n_tr, generator=g).to(dev)
        loss_sum, nb = 0.0, 0
        for s in range(0, steps_per_epoch * args.batch, args.batch):
            idx = perm[s:s + args.batch]
            if args.data == "crop36":
                tr = data["train"]
                img = to_float(tr["img"][idx], *norm)
                sz = tr["size"][idx]
                sz = torch.where(sz < 0, torch.randint(0, 3, sz.shape, generator=g).to(dev), sz)
                tpl = torch.randint(0, len(TEMPLATES), idx.shape, generator=g).to(dev)
                ids = table[tr["y"][idx], sz, tpl]
            else:
                img = to_float(data["train"]["img"][idx], *norm)
                ci = torch.randint(0, tr_ids.shape[1], idx.shape, generator=g).to(dev)
                ids = tr_ids[idx, ci]
            if args.sched == "cosine" and step >= args.warmup:
                total = args.epochs * steps_per_epoch
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, total - args.warmup)))
            else:
                lr = args.lr * min(1.0, (step + 1) / args.warmup)
            for pg in opt.param_groups:
                pg["lr"] = lr
            model.train()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.amp):
                ei, et = model.encode_image(img), model.encode_text(ids)
            loss = contrastive_loss(model, ei.float(), et.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            loss_sum += loss.item()
            nb += 1
            if not math.isfinite(loss.item()):
                print(f"NAN at epoch {ep} step {step}", flush=True)
                json.dump({"status": "nan", "epoch": ep}, open(os.path.join(out_dir, "final.json"), "w"))
                print(f"DONE {name} status=nan", flush=True)
                return
        extra = {"train_loss": loss_sum / nb, "epoch_sec": time.time() - t0, "lr": lr, "step": step}
        if dev.type == "cuda":
            extra["max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        if ep % eval_every == 0 or ep == args.epochs:
            rec = log_eval(ep, extra)
            sc = f" scale {rec['logit_scale']:.2f}" if "logit_scale" in rec else ""
            rc = f" recal {rec['recal_' + key]:.4f}" if args.eval_recal else ""
            print(f"ep {ep} loss {extra['train_loss']:.4f} {key} {rec[key]:.4f}{rc}{sc} {extra['epoch_sec']:.1f}s", flush=True)
    cf.close()
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    json.dump({"status": "ok", **rec}, open(os.path.join(out_dir, "final.json"), "w"), indent=1)
    print(f"DONE {name} status=ok {key}={rec[key]:.4f}", flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=["crop36", "coco"])
    p.add_argument("--loss", required=True, choices=["vse0", "vsepp", "clip", "siglip"])
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--sched", default="const", choices=["const", "cosine"],
                   help="const = [Unified] constant after warm-up; cosine = ablation (decay to 0 at the last step)")
    p.add_argument("--amp", action="store_true", help="bf16 autocast for the encoders")
    p.add_argument("--drop-last", action="store_true", default=True)
    p.add_argument("--probe-n", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=0, help="0 = epochs // 30 (at least 1)")
    p.add_argument("--init-img", default="", help="optional image-encoder state_dict (auxiliary run)")
    p.add_argument("--tag", default="")
    p.add_argument("--eval-recal", action="store_true",
                   help="also evaluate a BN-recalibrated copy at every eval (recal_* fields; training unchanged)")
    p.add_argument("--recal-batches", type=int, default=50)
    p.add_argument("--crop36-dir", default=os.path.expanduser("~/SSL_Wave1/datasets/ssl"))
    p.add_argument("--coco", default=os.path.expanduser("~/MM_Wave1/data/coco_sub/coco_sub.npz"))
    p.add_argument("--out", default=os.path.expanduser("~/MM_Wave1/runs/B"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
