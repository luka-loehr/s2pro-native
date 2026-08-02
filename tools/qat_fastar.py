#!/usr/bin/env python3
"""QAT self-distillation of the S2-Pro fast-AR decoder to group-wise INT4.

The BF16 fast-AR teaches its own 4-bit-quantized copy. No external data:
training records are (final-normed hidden, sampled semantic code) pairs
dumped by the serving engine (S2P_DUMP_FRAMES, src/slowar/slowar.c) during
ordinary generation; the exact BF16 teacher trajectory and logits are
recomputed here. The fake quantizer is a bit-faithful port of the engine's
load-time quantizer (s2p_intq_quant: symmetric 15 levels, groups of 32
along in-features, 32-step MSE clip search with every candidate rounded
through f16), so exporting the trained BF16 weights back into the
checkpoint and loading with S2P_INT4_ALL=1 deploys precisely the grid that
was trained.

Stages (all on one GPU):
  prep   — read the dump, recompute BF16-teacher greedy trajectories
  train  — teacher-forced KL over the 9 residual steps, STE fake-quant
  eval   — held-out per-step argmax agreement (teacher-forced + free-run)
  export — patch audio_decoder.* tensors into a copy of the checkpoint

Usage (inside a torch container, model dir + dump mounted):
  python3 qat_fastar.py --model /model --dump /data/frames.bin \
      --out /data/qat --steps 3000 --batch 256
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 2560
N_LAYER = 4
N_HEAD = 32
N_KV = 8
HEAD_DIM = 128
FFN = 9728
VOCAB = 4096
NCB = 10  # sequence positions 0..9; steps 1..9 predict codebooks 1..9
REC_BYTES = DIM * 2 + 4 + 10 * 4


# ---------------------------------------------------------------- quantizer
def fake_quant_g32(w: torch.Tensor) -> torch.Tensor:
    """Bit-faithful port of s2p_intq_quant (levels=7, G=32, MSE clip, f16
    scales). w [N, K] f32 -> dequantized f32, straight-through in train."""
    N, K = w.shape
    g = w.reshape(N, K // 32, 32)
    amax = g.abs().amax(dim=-1, keepdim=True)  # [N, K/32, 1]
    base = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    best_err = torch.full_like(amax, float("inf"))
    best_s = base.half().float()
    for c in range(32):  # candidate loop keeps memory at O(N*K)
        s = (base * (1.0 - c / 64.0)).half().float()
        q = torch.clamp(torch.round(g / s), -7, 7)
        err = ((g - q * s) ** 2).sum(dim=-1, keepdim=True)
        take = err < best_err  # strict <: first minimum, like the kernel
        best_err = torch.where(take, err, best_err)
        best_s = torch.where(take, s, best_s)
    qw = torch.clamp(torch.round(g / best_s), -7, 7) * best_s
    qw = torch.where(amax > 0, qw, g)  # all-zero group: kernel writes zeros
    return qw.reshape(N, K)


class FQLinear(nn.Module):
    """Linear whose weight passes through the deployment quantizer (STE)."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.float().clone())

    def forward(self, x):
        w = self.weight
        wq = w + (fake_quant_g32(w.detach()) - w.detach())
        return F.linear(x, wq)


class RefLinear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight.float().clone())

    def forward(self, x):
        return F.linear(x, self.weight)


class Int8Linear(nn.Module):
    """Deployment INT8 reference (per-out-channel absmax/127 RTN) — the
    quality bar the INT4 student must reach."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        w = weight.float()
        s = w.abs().amax(dim=1, keepdim=True) / 127.0
        s = torch.where(s > 0, s, torch.ones_like(s))
        self.register_buffer("weight",
                             torch.clamp(torch.round(w / s), -127, 127) * s)

    def forward(self, x):
        return F.linear(x, self.weight)


# ------------------------------------------------------------------- model
def precompute_freqs(base: float, device) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    t = torch.arange(NCB)
    f = torch.outer(t, freqs)
    cache = torch.stack([torch.cos(f), torch.sin(f)], dim=-1)
    return cache.bfloat16().float().to(device)  # bf16-rounded table


def rope(x: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    # x [B, T, H, D]; fc [T, D/2, 2]; interleaved pairs, butterfly in f32
    xs = x.float().reshape(*x.shape[:-1], -1, 2)
    f = fc.view(1, xs.size(1), 1, xs.size(3), 2)
    out = torch.stack(
        [xs[..., 0] * f[..., 0] - xs[..., 1] * f[..., 1],
         xs[..., 1] * f[..., 0] + xs[..., 0] * f[..., 1]], -1)
    return out.flatten(3).type_as(x)


class Block(nn.Module):
    def __init__(self, tensors, prefix, lin, eps):
        super().__init__()
        self.wqkv = lin(tensors[f"{prefix}.attention.wqkv.weight"])
        self.wo = lin(tensors[f"{prefix}.attention.wo.weight"])
        self.w1 = lin(tensors[f"{prefix}.feed_forward.w1.weight"])
        self.w3 = lin(tensors[f"{prefix}.feed_forward.w3.weight"])
        self.w2 = lin(tensors[f"{prefix}.feed_forward.w2.weight"])
        self.attn_norm = nn.Parameter(
            tensors[f"{prefix}.attention_norm.weight"].float().clone(),
            requires_grad=False)
        self.ffn_norm = nn.Parameter(
            tensors[f"{prefix}.ffn_norm.weight"].float().clone(),
            requires_grad=False)
        self.eps = eps

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def forward(self, x, fc, mask):
        B, T, _ = x.shape
        h = self.rms(x, self.attn_norm)
        qkv = self.wqkv(h)
        q, k, v = qkv.split([N_HEAD * HEAD_DIM, N_KV * HEAD_DIM,
                             N_KV * HEAD_DIM], dim=-1)
        q = rope(q.view(B, T, N_HEAD, HEAD_DIM), fc)
        k = rope(k.view(B, T, N_KV, HEAD_DIM), fc)
        v = v.view(B, T, N_KV, HEAD_DIM)
        k = k.repeat_interleave(N_HEAD // N_KV, dim=2)
        v = v.repeat_interleave(N_HEAD // N_KV, dim=2)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, N_HEAD * HEAD_DIM)
        x = x + self.wo(y)
        h = self.rms(x, self.ffn_norm)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class FastAR(nn.Module):
    def __init__(self, tensors, lin, eps, rope_base):
        super().__init__()
        self.embeddings = nn.Parameter(
            tensors["audio_decoder.embeddings.weight"].float().clone(),
            requires_grad=False)
        self.blocks = nn.ModuleList(
            Block(tensors, f"audio_decoder.layers.{i}", lin, eps)
            for i in range(N_LAYER))
        self.norm = nn.Parameter(
            tensors["audio_decoder.norm.weight"].float().clone(),
            requires_grad=False)
        self.output = lin(tensors["audio_decoder.output.weight"])
        self.register_buffer("fc", precompute_freqs(rope_base, "cpu"))
        self.eps = eps

    def forward_seq(self, hidden, sem, codes18):
        """Teacher-forced parallel forward. hidden [B,2560] f32, sem [B],
        codes18 [B,8] = c1..c8. Returns logits [B, 9, 4096] for steps 1..9."""
        B = hidden.shape[0]
        emb = self.embeddings
        seq = torch.empty(B, NCB, DIM, device=hidden.device)
        seq[:, 0] = hidden
        seq[:, 1] = emb[sem]
        seq[:, 2:] = emb[codes18]
        mask = torch.ones(NCB, NCB, dtype=torch.bool,
                          device=hidden.device).tril()
        x = seq
        fc = self.fc.to(hidden.device)
        for blk in self.blocks:
            x = blk(x, fc, mask)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) \
            * self.norm
        return self.output(x)[:, 1:]  # positions 1..9 predict cb 1..9

    @torch.no_grad()
    def greedy(self, hidden, sem):
        """Free-running 9-step greedy decode. Returns codes [B, 9]."""
        B = hidden.shape[0]
        dev = hidden.device
        emb = self.embeddings
        seq = torch.empty(B, NCB, DIM, device=dev)
        seq[:, 0] = hidden
        seq[:, 1] = emb[sem]
        out = torch.zeros(B, 9, dtype=torch.long, device=dev)
        fc = self.fc.to(dev)
        for j in range(1, NCB):
            T = j + 1
            mask = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
            x = seq[:, :T]
            for blk in self.blocks:
                x = blk(x, fc[:T], mask)
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) \
                * self.norm
            c = self.output(x[:, -1]).argmax(-1)
            out[:, j - 1] = c
            if j < NCB - 1:
                seq[:, j + 1] = emb[c]
        return out


# ------------------------------------------------------------------- data
def load_dump(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // REC_BYTES
    raw = raw[: n * REC_BYTES].reshape(n, REC_BYTES)
    hid = raw[:, : DIM * 2].copy().view(np.uint16)
    sem = raw[:, DIM * 2: DIM * 2 + 4].copy().view(np.int32).reshape(n)
    codes = raw[:, DIM * 2 + 4:].copy().view(np.int32).reshape(n, 10)
    hid_t = torch.from_numpy(hid.astype(np.uint16)).view(torch.bfloat16)
    return hid_t.reshape(n, DIM).float(), torch.from_numpy(sem).long(), \
        torch.from_numpy(codes).long()


def load_checkpoint_tensors(model_dir):
    from safetensors import safe_open
    tensors = {}
    files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    for f in files:
        with safe_open(os.path.join(model_dir, f), framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("audio_decoder."):
                    tensors[k] = sf.get_tensor(k)
    return tensors, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--holdout", type=int, default=8192)
    ap.add_argument("--dagger", type=float, default=0.5,
                    help="fraction of each batch forced with student "
                         "free-run codes in the second training half")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    adc = cfg.get("audio_decoder_config", cfg.get("audio_decoder", {}))
    eps = adc.get("norm_eps", 1e-6)
    rope_base = adc.get("rope_base", 1e6)
    print(f"[qat] norm_eps={eps} rope_base={rope_base}", flush=True)

    tensors, _ = load_checkpoint_tensors(args.model)
    print(f"[qat] {len(tensors)} audio_decoder tensors", flush=True)

    hid, sem, codes = load_dump(args.dump)
    n = hid.shape[0]
    print(f"[qat] {n} frames", flush=True)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(7))
    hold = perm[: args.holdout]
    train = perm[args.holdout:]

    teacher = FastAR(tensors, RefLinear, eps, rope_base).to(dev).eval()
    student = FastAR(tensors, FQLinear, eps, rope_base).to(dev)

    # ---- prep: exact BF16 teacher trajectories for ALL frames
    tpath = os.path.join(args.out, "teacher_codes.pt")
    if os.path.exists(tpath):
        tcodes = torch.load(tpath)
    else:
        outs = []
        with torch.no_grad():
            for i in range(0, n, 4096):
                h = hid[i: i + 4096].to(dev)
                s = sem[i: i + 4096].to(dev)
                outs.append(teacher.greedy(h, s).cpu())
                print(f"[qat] teacher gen {i + len(h)}/{n}", flush=True)
        tcodes = torch.cat(outs)
        torch.save(tcodes, tpath)
    agree_engine = (tcodes[:, :9] == codes[:, 1:10]).float().mean().item()
    print(f"[qat] teacher vs engine-dump code agreement: "
          f"{agree_engine:.4f}", flush=True)

    def eval_agreement(model, idx, tag):
        model.eval()
        tf_hits, fr_hits, cnt = 0, 0, 0
        with torch.no_grad():
            for i in range(0, len(idx), 2048):
                ix = idx[i: i + 2048]
                h = hid[ix].to(dev)
                s = sem[ix].to(dev)
                tc = tcodes[ix].to(dev)
                logits = model.forward_seq(h, s, tc[:, :8])
                tf_hits += (logits.argmax(-1) == tc).sum().item()
                fr = model.greedy(h, s)
                fr_hits += (fr == tc).sum().item()
                cnt += tc.numel()
        print(f"[qat] {tag}: teacher-forced argmax {tf_hits / cnt:.4f}, "
              f"free-run {fr_hits / cnt:.4f}", flush=True)
        return fr_hits / cnt

    int8_ref = FastAR(tensors, Int8Linear, eps, rope_base).to(dev).eval()
    eval_agreement(int8_ref, hold, "INT8 deployment bar")
    del int8_ref
    torch.cuda.empty_cache()
    base_fr = eval_agreement(student, hold, "INT4 pre-QAT")
    if args.eval_only:
        return

    # ---- train: teacher-forced KL
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    g = torch.Generator().manual_seed(11)
    student.train()
    for step in range(args.steps):
        ix = train[torch.randint(0, len(train), (args.batch,), generator=g)]
        h = hid[ix].to(dev)
        s = sem[ix].to(dev)
        tc = tcodes[ix].to(dev)
        force = tc[:, :8]
        if args.dagger > 0 and step * 2 >= args.steps:
            # DAgger phase (second half): for a slice of the batch, force
            # with the STUDENT's own free-run codes and let the teacher
            # provide the corrective targets on those prefixes — trains
            # recovery from the student's own drift (exposure bias).
            nd = int(args.batch * args.dagger)
            if nd > 0:
                with torch.no_grad():
                    student.eval()
                    sc = student.greedy(h[:nd], s[:nd])
                    student.train()
                force = torch.cat([sc[:, :8], tc[nd:, :8]], dim=0)
        with torch.no_grad():
            tlog = teacher.forward_seq(h, s, force)
        slog = student.forward_seq(h, s, force)
        loss = F.kl_div(F.log_softmax(slog, -1),
                        F.log_softmax(tlog, -1),
                        log_target=True, reduction="batchmean") / 9.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"[qat] step {step} loss {loss.item():.5f} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step % 500 == 499:
            eval_agreement(student, hold, f"INT4 step {step + 1}")
            student.train()

    fr = eval_agreement(student, hold, "INT4 post-QAT")
    print(f"[qat] free-run agreement {base_fr:.4f} -> {fr:.4f}", flush=True)

    # ---- export: patched audio_decoder tensors (bf16)
    patched = {}
    sd = student.state_dict()
    name_map = {"wqkv": "attention.wqkv", "wo": "attention.wo",
                "w1": "feed_forward.w1", "w3": "feed_forward.w3",
                "w2": "feed_forward.w2"}
    for i in range(N_LAYER):
        for short, full in name_map.items():
            patched[f"audio_decoder.layers.{i}.{full}.weight"] = \
                sd[f"blocks.{i}.{short}.weight"].bfloat16()
    patched["audio_decoder.output.weight"] = sd["output.weight"].bfloat16()
    from safetensors.torch import save_file
    save_file(patched, os.path.join(args.out, "qat_patch.safetensors"))
    print(f"[qat] wrote {len(patched)} patched tensors", flush=True)


if __name__ == "__main__":
    main()
