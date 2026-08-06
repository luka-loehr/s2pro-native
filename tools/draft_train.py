#!/usr/bin/env python3
"""EAGLE-style draft model for speculative multi-frame decoding.

The 36-layer slow-AR backbone is the per-frame cost; a draft that
predicts the backbone's NEXT final-normed hidden from (current hidden,
current frame) lets the engine propose k frames and verify them in one
batched backbone pass — on bandwidth-bound hardware a k-row verify pass
costs the same weight stream as a 1-row decode step, so accepted drafts
are nearly free frames.

Architecture (chosen so native inference reuses the engine's slow-AR
layer kernels 1:1):
  fuse:   Linear(2*2560 -> 2560, no bias) over [h_t ; e_{t+1}]
  layer:  ONE backbone-shaped block (GQA 32/8 head_dim 128 + qk-norm,
          SwiGLU ffn 9728), initialized from backbone layer 35
  norm:   final RMS norm (initialized from text_model.model.norm)
  head:   FROZEN tied lm-head slice (semantic rows + im_end row)

e_{t+1} is the backbone's own input embedding for position t+1, frozen
from the checkpoint: (embed[151678+sem_t] + sum_q cb_emb[q*4096 +
codes_t[q]]) / sqrt(11) — token embedding plus all ten codebook
embeddings, exactly src/slowar/slowar.c's decode path.

Training data: S2P_DUMP_FRAMES corpus generated at batch 1 (records are
per-take sequences, cut at the EOS sentinel with sem = -1).

Loss: SmoothL1 on the predicted hidden + KL of the frozen head's
semantic distribution (teacher || student). Metrics that matter for the
engine: semantic argmax agreement (greedy acceptance proxy) and the
fast-AR chain condition — greedy codes from the predicted hidden must
equal greedy codes from the true hidden (9/9), else a multi-frame chain
breaks at that position.

Usage (inside a torch container):
  python3 draft_train.py --model /model --dump /data/draft-corpus.bin \
      --out /data/draft --steps 4000 --batch 8 --seq 256
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qat_fastar import load_dump, FastAR  # noqa: E402

DIM = 2560
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
FFN = 9728
SEM_BASE = 151678
SEM_VOCAB = 4096
NCB = 10
ROPE_BASE = 1000000.0


def load_tensors(model_dir, names):
    from safetensors import safe_open
    out = {}
    for f in os.listdir(model_dir):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, f), framework="pt") as sf:
            for k in sf.keys():
                if k in names:
                    out[k] = sf.get_tensor(k)
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"missing tensors: {missing}")
    return out


def rope_cache(seq, device):
    freqs = 1.0 / (ROPE_BASE **
                   (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    t = torch.arange(seq)
    f = torch.outer(t, freqs)
    return torch.stack([torch.cos(f), torch.sin(f)],
                       dim=-1).bfloat16().float().to(device)


def rope(x, fc):
    xs = x.float().reshape(*x.shape[:-1], -1, 2)
    f = fc.view(1, xs.size(1), 1, xs.size(3), 2)
    out = torch.stack(
        [xs[..., 0] * f[..., 0] - xs[..., 1] * f[..., 1],
         xs[..., 1] * f[..., 0] + xs[..., 0] * f[..., 1]], -1)
    return out.flatten(3).type_as(x)


class DraftModel(nn.Module):
    """fuse + one backbone-shaped block + final norm."""

    def __init__(self, init, eps=1e-6):
        super().__init__()
        L = "text_model.model.layers.35."
        self.fuse = nn.Linear(2 * DIM, DIM, bias=False)
        with torch.no_grad():  # start as "pass the hidden through"
            self.fuse.weight.zero_()
            self.fuse.weight[:, :DIM] = torch.eye(DIM)
        self.wqkv = nn.Linear(DIM, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM,
                              bias=False)
        self.wo = nn.Linear(Q_HEADS * HEAD_DIM, DIM, bias=False)
        self.w1 = nn.Linear(DIM, FFN, bias=False)
        self.w3 = nn.Linear(DIM, FFN, bias=False)
        self.w2 = nn.Linear(FFN, DIM, bias=False)
        for name, key in [("wqkv", "attention.wqkv"), ("wo", "attention.wo"),
                          ("w1", "feed_forward.w1"),
                          ("w3", "feed_forward.w3"),
                          ("w2", "feed_forward.w2")]:
            getattr(self, name).weight.data.copy_(
                init[L + key + ".weight"].float())
        self.q_norm = nn.Parameter(init[L + "attention.q_norm.weight"].float())
        self.k_norm = nn.Parameter(init[L + "attention.k_norm.weight"].float())
        self.attn_norm = nn.Parameter(
            init[L + "attention_norm.weight"].float())
        self.ffn_norm = nn.Parameter(init[L + "ffn_norm.weight"].float())
        self.final_norm = nn.Parameter(
            init["text_model.model.norm.weight"].float())
        self.eps = eps

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def forward(self, h, e, fc, mask):
        """h,e [B,T,DIM] -> predicted next final-normed hidden [B,T,DIM]."""
        x = self.fuse(torch.cat([h, e], dim=-1))
        B, T, _ = x.shape
        a = self.rms(x, self.attn_norm)
        qkv = self.wqkv(a)
        q, k, v = qkv.split([Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM,
                             KV_HEADS * HEAD_DIM], dim=-1)
        q = q.view(B, T, Q_HEADS, HEAD_DIM)
        k = k.view(B, T, KV_HEADS, HEAD_DIM)
        v = v.view(B, T, KV_HEADS, HEAD_DIM)
        q = self.rms(q, self.q_norm)
        k = self.rms(k, self.k_norm)
        q, k = rope(q, fc[:T]), rope(k, fc[:T])
        k = k.repeat_interleave(Q_HEADS // KV_HEADS, dim=2)
        v = v.repeat_interleave(Q_HEADS // KV_HEADS, dim=2)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, T, Q_HEADS * HEAD_DIM)
        x = x + self.wo(o)
        f = self.rms(x, self.ffn_norm)
        x = x + self.w2(F.silu(self.w1(f)) * self.w3(f))
        return self.rms(x, self.final_norm)


def build_embeds(sem, codes, embed_main, cb_emb):
    """Frozen backbone input embedding for the NEXT position, from this
    frame: [N, DIM]."""
    e = embed_main[SEM_BASE + sem]
    for q in range(NCB):
        e = e + cb_emb[q * SEM_VOCAB + codes[:, q]]
    return e * (1.0 / math.sqrt(11.0))


def cut_takes(hid, sem, codes, min_len=8):
    takes, start = [], 0
    for i in range(sem.shape[0]):
        if sem[i].item() < 0:
            if i - start >= min_len:
                takes.append((start, i))
            start = i + 1
    if sem.shape[0] - start >= min_len:
        takes.append((start, sem.shape[0]))
    return takes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-weight", type=float, default=0.3)
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--holdout", type=int, default=48,
                    help="takes held out for eval (ignored with --eval-dump)")
    ap.add_argument("--eval-dump", default=None,
                    help="separate dump of UNSEEN-text takes; if set, all "
                         "of --dump trains and eval runs on this file")
    ap.add_argument("--feat-noise", type=float, default=0.0,
                    help="uniform noise magnitude added to input hiddens "
                         "during training (EAGLE-style augmentation)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--warm", default=None,
                    help="resume from a previous draft.safetensors")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    names = ["text_model.model.embeddings.weight",
             "audio_decoder.codebook_embeddings.weight",
             "text_model.model.norm.weight"]
    L = "text_model.model.layers.35."
    names += [L + k for k in
              ["attention.wqkv.weight", "attention.wo.weight",
               "attention.q_norm.weight", "attention.k_norm.weight",
               "attention_norm.weight", "ffn_norm.weight",
               "feed_forward.w1.weight", "feed_forward.w2.weight",
               "feed_forward.w3.weight"]]
    T = load_tensors(args.model, names)
    embed_main = T["text_model.model.embeddings.weight"].to(dev).float()
    cb_emb = T["audio_decoder.codebook_embeddings.weight"].to(dev).float()
    # frozen head slice: 4096 semantic rows (tied lm-head = embeddings)
    head_w = embed_main[SEM_BASE: SEM_BASE + SEM_VOCAB].contiguous()

    hid, sem, codes = load_dump(args.dump)
    takes = cut_takes(hid, sem, codes)
    rng = np.random.default_rng(23)
    if args.eval_dump:
        ev_hid, ev_sem, ev_codes = load_dump(args.eval_dump)
        ev_takes_ix = cut_takes(ev_hid, ev_sem, ev_codes)
        ev = (ev_hid, ev_sem, ev_codes, ev_takes_ix)
        tr_takes = takes
    else:
        if len(takes) < args.holdout + 16:
            raise SystemExit(f"only {len(takes)} takes; need more corpus")
        order = rng.permutation(len(takes))
        ev = (hid, sem, codes, [takes[i] for i in order[: args.holdout]])
        tr_takes = [takes[i] for i in order[args.holdout:]]
    n_frames = sum(b - a for a, b in tr_takes)
    print(f"[draft] corpus: train {len(tr_takes)} takes ({n_frames} "
          f"frames), eval {len(ev[3])} takes"
          f"{' (unseen texts)' if args.eval_dump else ''}", flush=True)

    model = DraftModel(T).to(dev)
    if args.warm:
        from safetensors import safe_open
        with safe_open(args.warm, framework="pt") as sf:
            sd = {k: sf.get_tensor(k).float() for k in sf.keys()}
        model.load_state_dict(sd)
        print(f"[draft] warm start from {args.warm}", flush=True)
    fc = rope_cache(args.seq + 1, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.05)
    mask = torch.tril(torch.ones(args.seq, args.seq, dtype=torch.bool,
                                 device=dev))

    def sample_batch():
        hs, es, ys = [], [], []
        for _ in range(args.batch):
            a, b = tr_takes[rng.integers(len(tr_takes))]
            T_take = b - a
            s = a if T_take <= args.seq + 1 else \
                a + int(rng.integers(T_take - args.seq - 1 + 1))
            e_end = min(b, s + args.seq + 1)
            h = hid[s:e_end].to(dev)
            sm = sem[s:e_end].to(dev)
            cd = codes[s:e_end].to(dev)
            e = build_embeds(sm, cd, embed_main, cb_emb)
            # inputs: (h_t, e from frame_t) t=0..T-2 -> target h_{t+1}
            hs.append(h[:-1])
            es.append(e[:-1])
            ys.append(h[1:])
        Tm = max(x.shape[0] for x in hs)
        def pad(lst):
            return torch.stack([F.pad(x, (0, 0, 0, Tm - x.shape[0]))
                                for x in lst])
        wm = torch.stack([F.pad(torch.ones(x.shape[0], device=dev),
                                (0, Tm - x.shape[0])) for x in hs])
        return pad(hs), pad(es), pad(ys), wm, Tm

    @torch.no_grad()
    def evaluate(step, fastar=None):
        model.eval()
        e_hid, e_sem, e_codes, e_takes = ev
        cos_s, arg_s, n = 0.0, 0, 0
        chain_ok, chain_n = 0, 0
        for a, b in e_takes:
            h = e_hid[a:b].to(dev)
            sm = e_sem[a:b].to(dev)
            cd = e_codes[a:b].to(dev)
            e = build_embeds(sm, cd, embed_main, cb_emb)
            hin, ein, y = h[:-1], e[:-1], h[1:]
            Tt = hin.shape[0]
            m = torch.tril(torch.ones(Tt, Tt, dtype=torch.bool, device=dev))
            with torch.autocast("cuda", torch.bfloat16):
                p = model(hin[None], ein[None],
                          rope_cache(Tt, dev), m)[0].float()
            cos = F.cosine_similarity(p, y, dim=-1)
            cos_s += cos.sum().item()
            arg_s += ((p @ head_w.T).argmax(-1) ==
                      (y @ head_w.T).argmax(-1)).sum().item()
            n += Tt
            if fastar is not None and chain_n < 512:
                k = min(Tt, 64)
                sel = torch.arange(k, device=dev)
                sm_next = sm[1:][sel]
                c_true = fastar.greedy(y[sel].bfloat16(), sm_next)
                c_pred = fastar.greedy(p[sel].bfloat16(), sm_next)
                chain_ok += (c_true == c_pred).all(dim=-1).sum().item()
                chain_n += k
        model.train()
        line = (f"[draft-eval] step {step}: hidden cos {cos_s / n:.4f}, "
                f"sem argmax {arg_s / n:.4f}")
        if chain_n:
            line += f", fastar chain 9/9 {chain_ok / chain_n:.4f}"
        print(line, flush=True)
        return arg_s / n

    fastar = None
    try:
        from qat_fastar import load_checkpoint_tensors, RefLinear
        ft, _ = load_checkpoint_tensors(args.model)
        fastar = FastAR(ft, RefLinear, 1e-6, ROPE_BASE).to(dev)
        fastar.eval()
        print("[draft] fast-AR chain metric enabled", flush=True)
    except Exception as ex:
        print(f"[draft] fast-AR chain metric unavailable: {ex}", flush=True)

    best = -1.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        h, e, y, wm, Tm = sample_batch()
        if args.feat_noise > 0.0:
            h = h + (torch.rand_like(h) * 2 - 1) * args.feat_noise
        m = mask[:Tm, :Tm]
        with torch.autocast("cuda", torch.bfloat16):
            p = model(h, e, fc, m).float()
            l_h = (F.smooth_l1_loss(p, y, reduction="none").mean(-1) *
                   wm).sum() / wm.sum()
            pl = (p @ head_w.T) / args.kl_temp
            yl = (y @ head_w.T) / args.kl_temp
            l_kl = (F.kl_div(F.log_softmax(pl, -1), F.log_softmax(yl, -1),
                             log_target=True, reduction="none").sum(-1) *
                    wm).sum() / wm.sum()
            loss = l_h + args.kl_weight * l_kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"[draft] step {step}/{args.steps} loss {loss.item():.4f} "
                  f"(h {l_h.item():.4f} kl {l_kl.item():.4f}) "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            acc = evaluate(step, fastar)
            if acc > best:
                best = acc
                from safetensors.torch import save_file
                save_file({k: v.bfloat16().contiguous()
                           for k, v in model.state_dict().items()},
                          os.path.join(args.out, "draft.safetensors"))
                print(f"[draft] saved best (sem argmax {best:.4f})",
                      flush=True)
    print(f"[draft] DONE best sem argmax {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
