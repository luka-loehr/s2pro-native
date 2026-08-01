# Porting Fish Audio S2-Pro to native Rust

> Authoritative implementation guide for a from-scratch native Rust + CUDA reimplementation of **Fish Audio S2-Pro (OpenAudio S2)**, using the reference PyTorch package (`fishaudio_s2_pro/…` + the SGLang-integrated modeling) as the source of truth. Every constant here is either read from source or from the checkpoint `config.json`; where a value is inferred or unverified it is flagged explicitly. Read this end-to-end before writing any kernel — several numerically load-bearing details (bf16 RoPE truncation, two-softmax sampling order, interleaved VQ injection) will silently produce garbled audio if gotten wrong.

---

## 1. Overview

S2-Pro is a **Dual-AR text-to-speech model**: a large autoregressive text/semantic model plus a small autoregressive acoustic-residual model, feeding a neural audio codec vocoder. It converts text (plus optional voice-reference audio for zero-shot cloning) into 44.1 kHz mono PCM.

### The three stages

```
                 STAGE 1 (CPU)              STAGE 2 (GPU, Dual-AR)                     STAGE 3 (GPU)
 text + refs ─▶ tokenize + VQ-encode ─▶ Slow-AR backbone (36L Qwen3) ─┐          ┌─▶ DAC / Firefly-GAN
                (Qwen3 ChatML prompt,    per frame: sample 1 semantic  │  [10×T]  │   from_indices ─▶ PCM
                 ref audio → codec        token from 155776 vocab      ├─ codes ─▶│   (final buffer OR
                 → 10×T VQ codes)                                      │          │    streaming windows
                                          Fast-AR decoder (4L)         │          │    w/ crossfade)
                                          per frame: argmax 9 residual ┘          └
                                          codebooks conditioned on
                                          the sampled semantic id
```

- **Stage 1 — Preprocess (CPU).** Build a hand-written Qwen3 ChatML prompt (NOT the Jinja template). Reference audio is encoded by the DAC codec to a `[10, T]` VQ-code stack; codebook-0 codes become real `<|semantic:i|>` token ids in the prompt, and all 10 codebooks ride alongside as a `vq_parts` side-channel fused as input embeddings.
- **Stage 2 — Dual-AR generate (GPU).** Autoregressive over **frames**. The **Slow-AR** backbone (36-layer dense Qwen3, `dim=2560`) samples exactly **one semantic token per frame** (codebook 0). The **Fast-AR** decoder (4-layer, same dims) then greedily emits the **9 residual codebook** values for that frame, conditioned on the Slow-AR's final-normed hidden state and the sampled semantic id. Per frame → 10 codec codes total (1 semantic + 9 residual).
- **Stage 3 — Vocoder (GPU).** Accumulated `[10, T]` code frames are decoded by a causal modded-DAC / Firefly-GAN codec to waveform, either as one final buffer or in overlapping streaming windows with crossfade.

### End-to-end tensor/dataflow

| Stage | Input | Output |
|---|---|---|
| Tokenize | text (UTF-8), refs (`text`, audio) | `input_ids [L] i64`, `vq_mask [L] bool`, `vq_parts: Vec<[10,Tᵢ] i32>` |
| Ref VQ-encode | ref audio `[T]` @ 44.1 kHz | `indices [10, T/2048] i64` |
| Slow-AR prefill | `input_embeds [L, 2560]` (VQ-injected) | hidden `[L, 2560]`, then per-frame decode |
| Slow-AR decode step | prev semantic token + prev-frame codes | logits `[155776]` bf16 → sampled `semantic_token` |
| Fast-AR (per frame) | final-normed hidden `[2560]` + `sem_id` | 9 residual codes (argmax) |
| Frame output | — | `output_codes[1:] = [sem_id, resid×9]` = `[10]` |
| Vocoder | codes `[10, T]` | PCM `[T × 2048]` @ 44100 Hz |

Frame rate is exactly **44100 / 2048 = 21.533203125 Hz** (≈46.44 ms/frame). Every code frame is 2048 audio samples.

---

## 2. Model dimensions & hyperparameters

All backbone/decoder values are **explicit in the checkpoint `config.json`** — do not re-derive them from the `configuration.py` defaults (which are generic placeholders and will produce a wrong model).

### Slow-AR backbone (`text_config`, `model_type = fish_qwen3`)

| Field | Value | Notes |
|---|---|---|
| `n_layer` | 36 | |
| `dim` (hidden) | 2560 | |
| `n_head` (query heads) | 32 | |
| `n_local_heads` (KV heads) | 8 | GQA 4:1 |
| `head_dim` | **128** | **EXPLICIT; ≠ dim/n_head (=80)** |
| Q proj width | 32×128 = **4096** | from hidden 2560 |
| K,V proj width | 8×128 = **1024** each | |
| fused QKV width | (32+8+8)×128 = **6144** | concat order q,k,v |
| `wo` | 4096 → 2560 | bias-free |
| `intermediate_size` (SwiGLU) | 9728 | |
| `attention_qk_norm` | **true** | per-head RMSNorm on Q & K |
| `attention_qkv_bias` / `attention_o_bias` | false / false | |
| `rope_base` | **1000000** (1e6) | |
| `norm_eps` (RMSNorm) | **1e-6** | all RMSNorms incl. QK-norm |
| `tie_word_embeddings` | **true** | lm_head = embedding `[155776, 2560]` |
| `vocab_size` | 155776 | |
| `max_seq_len` | 32768 | (runtime `context_length` = 4096) |
| `use_moe` | false | dense SwiGLU only; ignore MoE |
| dtype | **bfloat16** | (`use_bfloat16:false` is a stale HF flag — ignore) |

### Fast-AR audio decoder (`audio_decoder_config`, `model_type = fish_qwen3_audio_decoder`)

| Field | Value | Differs from backbone? |
|---|---|---|
| `n_layer` | 4 | ✓ |
| `dim` / `head_dim` / `n_head` / `n_local_heads` | 2560 / 128 / 32 / 8 | same |
| `intermediate_size` | 9728 | same |
| `attention_qk_norm` | **FALSE** | **✓ (no QK-norm)** |
| `tie_word_embeddings` | **FALSE** | **✓ (separate `output` head)** |
| `vocab_size` (= codebook_size) | 4096 | ✓ |
| `num_codebooks` | 10 | |
| `text_dim` | 2560 (== dim ⇒ `project_in` = Identity) | |
| `max_seq_len` | **11** (= num_codebooks + 1) | KV-cache depth |
| `rope_base` / `norm_eps` | 1e6 / 1e-6 | (per memory; verify in ckpt) |

Two embedding tables in the decoder module: **`codebook_embeddings`** = `Embedding(4096×10 = 40960, 2560)` (used only for Slow-AR reference-VQ injection), and **`embeddings`** = `Embedding(4096, 2560)` (the Fast-AR step input). Output head `output = Linear(2560, 4096, bias=False)`. `codebook_offsets = arange(10) * 4096 = [0, 4096, …, 36864]`.

### Codec (modded-DAC / Firefly-GAN, `configs/modded_dac_vq.yaml`)

| Field | Value |
|---|---|
| `sample_rate` | 44100 Hz, mono |
| `encoder_dim` | 64 |
| `encoder_rates` | `[2,4,8,8]` → `hop_length = 512` |
| `frame_length` | `hop_length × 4 = 2048` samples/token |
| token/frame rate | 44100/2048 = **21.533203125 Hz** |
| `latent_dim` | 64 × 2⁴ = **1024** |
| `decoder_dim` | 1536 |
| `decoder_rates` | `[8,8,4,2]` (prod 512) |
| decoder channels | 1536→768→384→192→96→1 |
| RVQ codebooks | **1 semantic (size 4096)** + **9 residual (size 1024)** = 10 streams |
| `codebook_dim` | 8 (low-dim projection per codebook) |
| RVQ downsample | `downsample_factor=[2,2]` (T ÷4, channels stay 1024) |
| encoder transformer | only deepest stage: 4 layers, dim 1024, 16 heads, window 512, causal |
| decoder transformer | **DEAD CODE** — built (`decoder_transformer_layers=[4,0,0,0]`) but commented out of forward; do NOT implement |
| RVQ pre/post module | `WindowLimitedTransformer`, 8 layers, dim 1024, 16 heads, head_dim 64, intermediate 3072, window 128, `rope_base=10000`, `norm_eps=1e-5`, causal (two independent weight sets: `quantizer.pre_module.*`, `quantizer.post_module.*`) |
| `causal` | true (all convs are causal) |

`codes` tensor: `[B, 10, T] int64`, order `[semantic(1), residual(9)]`; semantic ∈ 0..4095, residual ∈ 0..1023.

---

## 3. Tokenizer & prompt format

Stock **Qwen3 byte-level BPE** loaded from `model/tokenizer.json` (`tokenizers` crate). Pipeline: **NFC normalize → Split on the isolated regex → ByteLevel(add_prefix_space=false, use_regex=false) → BPE(151387 merges, no byte_fallback, no ignore_merges) → ByteLevel decode**. No automatic BOS/EOS (`add_bos_token=false`, post-processor is ByteLevel only). `clean_up_tokenization_spaces=false`.

Pre-tokenizer regex (replicate **verbatim**; ignore the typo'd `FISH_TIKTOKEN_PATTERN` in `fish_speech/tokenizer.py`):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

### Special / semantic token ids (stable in `tokenizer.json`)

| Token | Id |
|---|---|
| base BPE vocab | 151643 tokens |
| `<|im_start|>` | 151644 |
| `<|im_end|>` (real EOS/stop) | **151645** |
| `<|pad|>` | 151669 |
| `<|voice|>` (assistant modality marker) | 151673 |
| `<|audio_pad|>` | 151677 |
| `<|semantic:i|>` | **151678 + i** (i = 0..4095, contiguous, verified linear) |
| `semantic_start` / `semantic_end` | 151678 / 155773 |
| total vocab | **155776** |

`<|semantic:i|>` ↔ global id `151678 + i`; codec logits 0..4095 map to global ids by `+151678`. Ids 155774/155775 exist above `semantic_end` (unused specials).

**`<|speaker:N|>` is NOT a special token** — it is ordinary UTF-8 text, byte-level-BPE'd into multiple sub-word ids. Registering it as one id will break parity. Do **not** register it. Register the Qwen3 base specials (151643..151645), the Fish specials (151669..151677), and all 4096 semantic tokens as non-normalized special added tokens.

### Exact prompt assembly (`build_prompt`, per-segment tokenization)

Each `append_text` call runs `tokenizer.encode()` on that **literal string independently** — BPE merges never cross a segment boundary. Reproduce these exact literals (including newlines):

**If references present** (system block, one block for ALL refs combined):
1. `<|im_start|>system\n`
2. `convert the provided text to speech reference to the following:\n\nText:\n`
3. per ref: `<|speaker:{speaker}|>{ref.text}` (only if `ref.text` non-empty); collect `ref.vq_codes`
4. `\n\nSpeech:\n`
5. `append_vq(torch.cat(all_codes, dim=1))` — time-concatenate every ref's `[10,T]` along axis 1 **first**, then encode
6. `<|im_end|>\n`

**User block:**
7. `<|im_start|>user\n`
8. `<|speaker:{speaker}|>{text}`
9. `<|im_end|>\n`

**Assistant open** (generation begins immediately after `<|voice|>`, no trailing newline):
10. `<|im_start|>assistant\n<|voice|>`

`append_vq(codes)` internals: token ids = `<|semantic:{c}|>` for each `c` in **codebook-0 only** (`codes[0]`); vq_mask segment all-True; the **full `[10,T]`** tensor pushed to `vq_parts`.

**Two injection channels from one reference:** codebook-0 → real semantic token ids in `input_ids` (mask=True); ALL 10 codebooks → `vq_parts`, fused as input embeddings at masked positions. Both must be produced. Multiple refs share exactly ONE `\n\nSpeech:\n` header and ONE semantic block.

Downstream: `input_ids → i64`, `vq_mask → bool`, request `vocab_size = len(tokenizer) = 155774` (NOT the base `tokenizer.vocab_size=151643`, and NOT the model logits/embedding width 155776), `stop_token_ids = [151645]`. Radix-cache `extra_key` = blake2b-128 over **all** codebooks (shape-string + int32 LE bytes) so refs sharing cb0 but differing cb1..9 key distinctly. `top_k` constrained to `-1` or `[1,30]`.

---

## 4. Slow-AR forward & VQ embedding injection

> The per-layer forward op-order below is inferred from config flags + gpt-fast convention; it lives in the SGLang modeling code (`S2ProSGLangTextModel`) and should be confirmed there. Config values and RoPE math are ground-truth.

**Per Slow-AR layer:**
```
h  = x + wo( SDPA_GQA( rope( qk_norm( split(wqkv(rmsnorm(x))) ) ) ) )   # qk_norm on Q,K before RoPE
out = h + w2( silu(w1(rmsnorm(h))) * w3(rmsnorm(h)) )                    # SwiGLU, intermediate 9728
```
After 36 layers: final RMSNorm → tied `lm_head` (`hidden @ embed.weight^T`) → logits `[155776]` bf16.

- Fused `wqkv` (width 6144) split to q=4096 / k=1024 / v=1024; reshape q→`[.,32,128]`, k,v→`[.,8,128]`.
- **Per-head QK-RMSNorm** (eps 1e-6) applied on the 128-dim head vector, **before** RoPE. Separate `q_norm`/`k_norm` weights per layer — locate in checkpoint.
- GQA: replicate each of 8 KV heads 4× to serve 32 Q heads. Causal mask. Scale `1/sqrt(128)`.
- SwiGLU: `down(silu(gate(x)) * up(x))`, `silu(x)=x·σ(x)`, intermediate 9728, no bias.
- Weight remap (checkpoint → module): `self_attn.o_proj → attention.wo`, `input_layernorm → attention_norm`, `post_attention_layernorm → ffn_norm`, `gate_up_proj` shard0/shard1 → `w1`/`w3`, `down_proj → w2`, `embed_tokens → embeddings`.

### VQ embedding injection (interleaved reference codes)

Two identical-scale injection sites, both dividing by **`1/sqrt(num_codebooks+1) = 1/sqrt(11) ≈ 0.30151`**:

**Prefill** (`embed_text_dim`, `_build_prefill_input_embeds`): at every `vq_mask=True` (semantic) position, replace the token embedding with
```
(text_embed + Σ_{c=0..9} codebook_embeddings(vq_parts[c] + codebook_offsets[c])) / sqrt(11)
```
The sum spans all 10 codebooks (via `codebook_offsets = arange(10)*4096`) plus the base text embedding, then scaled.

**Decode** (`forward`, per step): `before_decode` loads the **previous frame's** 10 codebook values into `_vq_codes[B,10]` and sets `_vq_mask` from whether the current input token is semantic. When semantic:
```
hidden = (embed_tokens(next_semantic_token) + Σ codebook_embeddings(_vq_codes + offsets)) * (1/sqrt(11))
```

`codebook_offsets` and `freqs_cis` are `persistent=False` buffers (absent from the checkpoint) — rematerialize them on the real device at load, or they contain garbage that overflows the embedding.

---

## 5. Semantic sampling algorithm (EXACT order) + residual argmax

The crown jewel, in `_decode_codebooks`. **All sampling math runs in float32 after one bf16 round.** Reordering any step changes outputs.

**Constants:** `_GRAPH_TOP_K=30`, `_DEFAULT_TEMPERATURE=0.8`, `_DEFAULT_TOP_P=0.8`, `_DEFAULT_REP_PENALTY=1.1`, `rep_history_len=16`, `ras_temperature=1.0`, `ras_top_p=0.9`, `_ras_range=arange(4,0,-1)=[4,3,2,1]`, `_NO_SEED=-1`, `_vq_scale=1/sqrt(11)`. `_semantic_bias[155776]` bf16 = `0.0` on `[151678..155773]` (4096 ids) **and** on `im_end` (151645); `-inf` everywhere else → exactly **4097** unmasked candidates.

**Exact ordering:**

1. **Bias:** `biased = logits + _semantic_bias`.
2. **Precision truncation:** `biased = biased.to(bf16).to(float32)`. (Round `(logits+bias)` through bf16 *before* widening to f32 — bit-exactness depends on this.)
3. **RAS detect:** gather last 4 emitted tokens via `count - [4,3,2,1]` (clamped ≥0); `sort` + adjacent-equal to detect **any** duplicate among the 4. `use_ras = has_dup AND count ≥ 4`. If set for a row, that row's `temperature ← 1.0`, `top_p ← 0.9`. **`top_k` and `rep_penalty` are unchanged under RAS.**
4. **Repetition penalty** (on the RAW biased f32 logits, BEFORE top-k/top-p): gather scores at `_prev_tokens`; `penalized = score * 1.1 if score < 0 else score / 1.1`; apply only at valid positions (`pos < count`, using capped `_prev_token_count`); `scatter_` back by token id.
5. **Top-k:** `topk(biased, 30)` → `top_k_logits`, `top_k_indices` (descending-logit order). Mask ranks `≥ effective_k` (= `clamp(top_k, max=30)`) to `-inf`.
6. **Top-p (UN-temperatured):** `cum = cumsum(softmax(top_k_logits))` (NO temperature here); mask `cum > top_p` to `-inf`, but **always keep rank 0**.
7. **Temperature softmax:** `probs = softmax(top_k_logits / max(temperature, 1e-5))`. **Temperature applied HERE only, second softmax, after filtering.**
8. **Sample:** unseeded → `torch.multinomial(probs)` (always run for whole batch); seeded → `multinomial_with_seed(log(probs), seed, step_count)`; pick per-row by `seed ≥ 0`. `semantic_token = top_k_indices.gather(choice)`.

Two distinct softmaxes (nucleus without temperature, final with) — folding temperature into top-p diverges. Two counters: **capped** `_prev_token_count` (≤16) for rep-penalty/history validity; **uncapped** `step_count` for seeded sampling — do not conflate.

**Residual codebooks: pure argmax** (deterministic, over bf16 logits). Only the semantic token is stochastic. Because the residual cascade is greedy over bf16 near-ties, free-running bit-exactness is unattainable — **gate parity on teacher-forced per-step argmax agreement (~95–97%)**, as the reference does.

---

## 6. Fast-AR decode loop (per-frame 9-residual generation)

Driven by `_decode_codebooks` after the semantic token is sampled. The Fast-AR "sequence" dimension is the **codebook index (0..9)**, not time. KV cache depth 11, 8 KV heads × 128 head_dim, bf16, **zeroed each frame** (`reset_caches()`).

```
reset_caches()                                        # zero KV cache
fast_input = project_in(hidden)   # Identity → = final-NORMED backbone hidden, [B,1,2560]
forward_kvcached(fast_input, codebook_idx=0)          # PRIME KV pos 0 — LOGITS DISCARDED
sem_id  = clamp(semantic_token - semantic_begin_id, 0, 4095)   # forced 0 if semantic_token == im_end
cb_hidden = embeddings(sem_id).unsqueeze(1)           # [B,1,2560], the 4096-vocab table
output_codes[:,0] = semantic_token ; output_codes[:,1] = sem_id
for cb_idx in 1..=9:                                  # 9 residual steps
    cb_logits = forward_kvcached(cb_hidden, cb_idx)[:, 0, :4096]   # fills KV pos cb_idx
    cb_token  = argmax(cb_logits)                     # DETERMINISTIC, no sampling
    cb_hidden = embeddings(cb_token).unsqueeze(1)
    output_codes[:, cb_idx+1] = cb_token
```

Inside `forward_kvcached(x, codebook_idx)`: `input_pos.fill_(codebook_idx)`; `freqs = freqs_cis[input_pos]`; `cache_seqlens = codebook_idx (int32)`; for each of 4 layers `h = x + attn(attention_norm(x), freqs, cache_seqlens); x = h + ff(ffn_norm(h))`; return `output(norm(x))`. Attention: `wqkv → split q=4096/k=1024/v=1024 → reshape → (qk_norm SKIPPED) → rope(q), rope(k) → append k,v at cache_seqlens, attend causally over cache[:idx+1] with GQA ×4 → wo`. Scale `1/sqrt(128)`.

**Key facts:** codebook-0 is a **priming pass only** (its logits are discarded; the semantic code comes from the Slow-AR). The RoPE table has only 10 rows (`[10,64,2]` bf16); pos 10 of the 11-deep cache is allocated-but-unused. The GB10 fallback `flash_attn_kvcache_op` runs SDPA in **float32** then casts to bf16 (the original Hopper path used bf16 flash-attn — pick the target you must match). `output` is sliced to `[:4096]` before argmax (no-op here, but don't assume width > codebook_size).

**Frame output to codec:** `output_codes[1:] = [sem_id, resid_1..9]` = 10 rows. Column 0 (raw `semantic_token`) is bookkeeping/EOS only. `semantic_token == im_end` ends generation; that frame's codes are computed but **dropped** downstream.

---

## 7. DAC vocoder

Decode path is `DAC.from_indices(indices[B,10,T])` = `decoder(quantizer.decode(indices))`.

**`DownsampleResidualVectorQuantize.decode`** (mutates indices in place — clone before reuse):
1. `clamp` `indices[:,0] ≤ 4095` and `indices[:,1:] ≤ 1023` (in place).
2. `z_semantic = semantic_quantizer.from_codes(indices[:,:1])` (codebook 0, size 4096).
3. `z_residual = quantizer.from_codes(indices[:,1:])` (9 codebooks, size 1024).
4. `z = z_semantic + z_residual`.
5. `post_module` transformer (8-layer causal, window 128, dim 1024).
6. `upsample` (CausalTransConv k=stride=factor + ConvNeXtBlock) back to 4× length; `decode()` **returns this directly with NO length reconciliation**. The left-pad / left-crop-to-`original_shape` logic exists **only in `forward()`**, not in the decode/vocode path — do NOT add it here (the decode path has no "original T" to restore to).

**`Decoder`:** `conv(1024→1536, k7)` → 4 DecoderBlocks (each: `Snake1d` → causal transposed-conv up by stride, halving channels → 3 ResidualUnits dil 1/3/9), channels 1536→768→384→192→96 → `Snake1d` + `conv(96→1, k7)` + `Tanh` → PCM `[B,1,T×2048]`.

**Encode path** (voice cloning only, `DAC.encode`): audio `[B,T]` → `unsqueeze(1)` → right-pad with zeros to `ceil(T/2048)*2048` → `Encoder` → `quantizer.forward` → `codes = cat([semantic(1), residual(9)], dim=1)` `[B,10,T/2048]`; `indices_lens = ceil(audio_lengths/2048)`.

### Streaming windowing (`build_stream_vocoder_chunk`)

Bounded-window O(n) decode with crossfade. Defaults: `stream_stride=10`, `stream_followup_stride=90`, `stream_overlap_tokens=20`, `stream_crossfade_samples=512`, `samples_per_token = frame_length = 2048`.

- Accumulate code chunks; `total_tokens += chunk_width`. Return `None` until `total_tokens` crosses `next_vocode_tokens` (first threshold = `stream_stride=10`; subsequent = `+stream_followup_stride=90`).
- Decode an **overlapping** window: `window_start = max(code_start, emitted - overlap)`; `codebook_codes = window_codes[1:]`; `audio = codec.from_indices(codebook_codes[None])`.
- Drop `overlap_samples = overlap_token_count × 2048` (re-decoded to preserve codec causal context); `delta = audio[overlap_samples:]`.
- Crossfade `delta` with `pending_tail` (linspace fade, hold last 512 samples as new tail); emit delta; trim retained codes to `keep_from = max(0, total - overlap)`.
- `on_stream_done` flushes remaining + `pending_tail`, then runs the full `_vocode_payloads` for the terminal result.

**Final (non-streaming) path** (`_vocode_payloads`): drop row 0 (`output_codes[1:]` = 10 rows), pad to max_len, stack, `codec.from_indices`, slice each to `length × 2048` samples.

---

## 8. Known fidelity pitfalls

These are the things that produce garbled audio if gotten wrong. Be paranoid about all of them.

1. **bf16 RoPE precision truncation (everywhere).** Every RoPE cos/sin table in this model is computed in fp32 then **rounded to bf16 before use**:
   - Backbone: `bootstrap.truncate_rope_to_bf16` rounds the cache bf16→fp32→apply.
   - Fast-AR & codec: `precompute_freqs_cis` returns `.to(bfloat16)`.
   `apply_rotary_emb` upcasts activations to f32, multiplies by the **bf16-rounded** cos/sin, then casts back. A naive full-fp32 RoPE table will NOT bit-match. Recipe: build freqs in f32 → round to bf16 → store (cos,sin) pairs → do the butterfly in f32 accumulation → round result to bf16.

2. **Interleaved (gpt-fast / GPT-J) RoPE layout, NOT rotate-half.** Pairs are **adjacent** `(2i, 2i+1)` via `reshape(...,-1,2)`. `x_out = [x0·cos − x1·sin, x1·cos + x0·sin]` flattened back. `head_dim=128 → 64 pairs` (backbone/fast-AR); codec `head_dim=64 → 32 pairs`. Using the HF/Llama `rotate_half` (split-half) convention is silently wrong. `is_neox_style=False` on both AR heads.

3. **Semantic-logit precision truncation before sampling.** `biased = (logits + semantic_bias).to(bf16).to(fp32)` — round the sum through bf16 first. All subsequent sampling math is f32.

4. **Two softmaxes / temperature ordering.** Top-p nucleus uses an **un-temperatured** softmax; only the final probs divide by `temperature.clamp(min=1e-5)`. Temperature is applied **after** top-k and top-p, in a separate softmax. Folding temperature earlier changes which tokens are in the nucleus. Full order: `bias → bf16 round → RAS select → rep-penalty(scatter) → top-k(30) → top-p(rank0 kept) → temperature softmax → multinomial`.

5. **Fast-AR must be primed on the FINAL-NORMED backbone hidden** (post `self.norm`), passed through `project_in` (Identity) — **NOT** the raw pre-norm last-layer residual. This is the single most common porting bug (the "P2 fix" in project memory).

6. **Codebook-0 forward is priming only** — its logits are discarded; codebook 0 (semantic) is sampled from the Slow-AR, never predicted by the Fast head. Consuming cb0 logits corrupts everything.

7. **Interleaved-VQ-injection correctness.** Two DIFFERENT embedding tables: `codebook_embeddings[40960,2560]` (VQ combination, via `+codebook_offsets`) vs `embeddings[4096,2560]` (Fast-AR seeding). Do not conflate. Scale is exactly `1/sqrt(11)` at BOTH prefill and decode, over all 10 codebook embeds + base token embed. `vq_mask` alignment: only masked/semantic positions get their embedding overridden.

8. **QK-norm asymmetry.** Backbone `attention_qk_norm=TRUE` (per-head RMSNorm eps 1e-6 on Q,K before RoPE); Fast-AR `attention_qk_norm=FALSE`. Easy to accidentally add QK-norm to the fast head.

9. **RMSNorm semantics.** Match `nn.RMSNorm`: `y = x · rsqrt(mean(x², last) + eps) · weight`, **eps inside the sqrt**. The `MyRMSNorm` fallback (only under `FISH_BATCH_INVARIANT`) uses a DIFFERENT formula (eps outside, L2-norm) — do not port it unless deliberately matching batch-invariant mode. Codec RMSNorm eps = 1e-5; AR eps = 1e-6.

10. **Output-codes column layout.** `[semantic_token(raw vocab id), sem_id(0..4095), cb1..cb9]`. Codec consumes rows `[1:]` = `[sem_id, resid×9]`. Feeding col 0, or dropping the wrong row, corrupts audio.

11. **Codec causal-conv manual padding.** All padding is manual; inner `nn.Conv1d` has `padding=0`; the `padding=` kwarg to `CausalConvNet` is silently ignored. Left pad = `(kernel-1)*dilation+1 - stride` + dynamic right pad (`get_extra_padding_for_conv1d`), `mode='constant'` value 0 (NOT reflect). ResidualUnit residual is **right-cropped** (`x[...,:-pad]`) when causal. Reproduce or lengths drift.

12. **Codec dead decoder transformer.** `decoder_transformer_layers=[4,0,0,0]` instantiates a transformer **as a local variable** in `DecoderBlock.__init__`, but it is commented out of the `nn.Sequential` and **never registered as a submodule** — so its parameters are **absent from the model state_dict** and it never runs. Do not implement it in decode.

13. **Residual head width mismatch.** Fast-AR output is 4096-wide and argmax is over `[:4096]`, but codec residual codebooks are only 1024 entries. Trained weights keep residual argmax in 0..1023; do **not** mask to 1024 (would diverge). The codec `decode` clamps `indices[:,1:] ≤ 1023` as a safety net.

14. **Weight-norm must be folded at load.** The checkpoint stores parametrized (g, v) weights for all `WNConv1d`/`WNConvTranspose1d`/CausalConv; materialize `g · v/‖v‖` into plain conv weights.

---

## 9. Weights

Weights are **gated under the Fish Audio Research License** and are **NOT included in this repo**. Obtain the checkpoint separately; do not commit it.

### Checkpoint layout

| Group | Tensors | dtype |
|---|---|---|
| **Slow-AR backbone** | `embed_tokens.weight [155776,2560]` (tied → lm_head), per-layer `wqkv`/`wo`/`q_norm`/`k_norm`/`attention_norm`/`ffn_norm`/`w1`/`w2`/`w3`, final `norm` | bf16 |
| **Fast-AR** (`audio_decoder.*`) | `codebook_embeddings [40960,2560]`, `embeddings [4096,2560]`, 4× layer weights (no q/k_norm), `norm`, `output.weight [4096,2560]` (untied) | bf16 |
| **Codec** (`modded-dac-msstftd-step-1380000.pth`, loaded `strict=False`) | `encoder.*`, `decoder.*`, `quantizer.{downsample,pre_module,semantic_quantizer,quantizer,post_module,upsample}.*` — weight-normed convs (parametrized), transformer weights | (loaded, weight-norm folded at load) |

Notes: the Fast-AR checkpoint contains **NO** `q_norm`/`k_norm` tensors (its `attention_qk_norm=false`, so those submodules are never constructed, and the decoder loads `strict=True` — the keys are absent). Only the backbone (`qk_norm=true`) carries `q_norm`/`k_norm`. `codebook_offsets` and `freqs_cis` are `persistent=False` — rematerialize at load. The external `descript-audio-codec` package (`dac.nn.quantize`, `dac.nn.layers`) supplies `ResidualVectorQuantize`/`VectorQuantize` (per-codebook `in_proj 1024→8`/`out_proj 8→1024`, L2-normalized code distance, `from_codes`) and `Snake1d` — these are NOT vendored; `pip install descript-audio-codec` and port them faithfully. Authoritative dtype for the AR models is bf16 (`use_bfloat16:false` is a stale flag).

---

## 10. Suggested Rust crate/module structure

| Crate | Maps to | Responsibility |
|---|---|---|
| `s2pro-tokenizer` | `tokenizer.py`, `fish_speech/tokenizer.py`, `request_builders.py` | Qwen3 BPE (via `tokenizers`), `build_prompt` literal assembly, semantic-id mapping, vq_mask + vq_parts, blake2b cache key |
| `s2pro-slow-ar` | `sglang_model.py` (backbone), `fish_qwen3` config, `utils.py` RoPE | 36-layer dense Qwen3 backbone: fused QKV, QK-RMSNorm, interleaved bf16 RoPE, GQA SDPA, SwiGLU, tied lm_head, VQ embedding injection |
| `s2pro-sampling` | `_decode_codebooks` sampling half | f32 constrained sampler: bias, bf16 round, RAS, rep-penalty, top-k/top-p, two-softmax temperature, seeded/unseeded multinomial |
| `s2pro-fast-ar` | `audio_decoder.py` | 4-layer residual decoder, per-frame 10-step KV-cached loop (1 prime + 9 argmax), two embedding tables, `embed_text_dim` VQ combine |
| `s2pro-codec` | `modded_dac.py`, `rvq.py`, `modded_dac_vq.yaml` (+ external DAC) | Causal DAC encoder/decoder, DownsampleRVQ (semantic+residual, from_codes), WindowLimitedTransformer, ConvNeXt, Snake1d, weight-norm folding |
| `s2pro-vocoder-stream` | `streaming_vocoder.py` | Streaming window scheduler: stride/overlap/crossfade, pending-tail, final batch vocode |
| `s2pro-runtime` | `config.py`, `payload_types.py`, `model_runner.py`, `stages.py`, `bootstrap.py` | Pipeline orchestration, per-session decode state buffers, prefill embed builder, EOS/harvest, RoPE bf16 truncation, buffer rematerialization |
| `s2pro-cuda` (csrc) | GPU kernels | Attention, RMSNorm, RoPE butterfly, SwiGLU, codec convs, sampling — the numeric hot paths |

Persistent per-session decode state to mirror (mutated in place each step): `_vq_codes[B,10]`, `_vq_mask[B]`, `_output_codes[B,11]`, `_output_semantic_ids[B]`, `_semantic_bias[155776]` bf16, sampling tensors, `_prev_tokens[B,16]` + count. Runtime knobs from the reference: `dtype=bf16`, `disable_cuda_graph=True` (backbone), Fast-AR layers torch-compiled, `attention_backend=triton`, `context_length=4096`, `max_new_tokens` clamped to `4096-1-prompt_len`.

---

## 11. Per-file map of the reference package (`fishaudio_s2_pro/…`)

Canonical root: the `fishaudio_s2_pro/` reference package from the SGLang-Omni day-0 S2-Pro integration (obtain from upstream; not vendored here).

| File | Purpose |
|---|---|
| `config.py` | `S2ProPipelineConfig` — declares the fixed 3-stage pipeline (preprocess → tts_engine → vocoder), `stream_to`, device, `max_new_tokens=2048`, `architecture=FishQwen3OmniForCausalLM`. |
| `payload_types.py` | `S2ProState` dataclass — per-request pipeline state (input_ids, vq_mask_tokens, vq_parts, num_codebooks=10, codebook_size=4096, sampling params, output_codes, sample_rate=44100). |
| `stages.py` | Stage-1 body `_preprocess` — normalize inputs, encode references to VQ (cache-backed), call `build_prompt`, pack `S2ProState`. |
| `tokenizer.py` | `S2ProTokenizerAdapter.build_prompt` + `_InferencePromptEncoder` — the exact ChatML assembly, `append_text`/`append_vq`/`finish`, semantic-id mapping, eos/semantic_begin/end properties. |
| `request_builders.py` | `build_sglang_tts_request`, `_ref_vq_fingerprint` (blake2b-128 over all codebooks), `validate_s2pro_top_k`, `stream_output_builder`. Sets `vocab_size=len(tokenizer)=155774`, `stop_token_ids=[im_end]`. |
| `model_runner.py` | `FishS2ProModelRunner` — `_build_prefill_input_embeds` (VQ injection), `before_decode` (sync `_vq_mask`/`_vq_codes`), `collect_s2pro_step_outputs` (EOS handling, frame harvest, RAS history). |
| `sglang_model.py` | `S2ProSGLangTextModel` — **the numerical core**: `forward` (backbone + VQ combine + tied logits), `_decode_codebooks` (constrained sampling + Fast-AR loop), `setup_vq_decode` (all persistent buffers/constants). |
| `streaming_vocoder.py` | `build_stream_vocoder_chunk` (overlapping-window streaming decode + crossfade) and `S2ProVocoderScheduler._vocode_payloads` (final batch vocode). |
| `bootstrap.py` | Setup: `truncate_rope_to_bf16` (backbone RoPE bf16 rounding), `_rematerialize_audio_decoder_buffers` (rebuild persistent=False buffers on real device). |
| `configuration.py` | HF config classes: `FishQwen3Config`, `FishQwen3AudioDecoderConfig`, `FishQwen3OmniConfig` — special/semantic token ids, dimension asserts, `AutoConfig.register`. Defaults are placeholders; real values in `config.json`. |
| `fish_speech/tokenizer.py` | Token string constants: `SEMANTIC_TOKEN_TEMPLATE='<|semantic:{i}|>'`, `IM_START/IM_END/MODALITY_VOICE`, the documentation-only `FISH_TIKTOKEN_PATTERN` (has an escaping typo — NOT the serving regex). |
| `fish_speech/models/text2semantic/audio_decoder.py` | `FishQwen3AudioDecoder` — Fast-AR module: `forward_kvcached`, `embed_text_dim`, `setup_caches`, `Attention`, `flash_attn_kvcache_op`, `FeedForward`, RMSNorm alias, two embedding tables, output head. |
| `fish_speech/models/text2semantic/utils.py` | `precompute_freqs_cis` / `apply_rotary_emb` (interleaved gpt-fast RoPE, bf16 table), `find_multiple`. |
| `fish_speech/models/dac/modded_dac.py` | Codec: `DAC` (encode/decode/from_indices), `Encoder`/`EncoderBlock`, `Decoder`/`DecoderBlock`, `ResidualUnit`, `WindowLimitedTransformer`, `Attention`, causal conv nets, RoPE, RMSNorm. |
| `fish_speech/models/dac/rvq.py` | `DownsampleResidualVectorQuantize` (downsample → pre_module → semantic + residual quantize → post_module → upsample), `.decode` (in-place clamp + from_codes), ConvNeXtBlock. Imports external `ResidualVectorQuantize`. |
| `configs/modded_dac_vq.yaml` | Codec hyperparameters: sample_rate, encoder/decoder rates, RVQ config, pre/post transformer module (shared anchor), downsample factors. |
| `model/config.json` | **Authoritative** backbone + audio-decoder hyperparameters and special token ids. |
| `model/tokenizer.json`, `tokenizer_config.json` | Qwen3 byte-level BPE model + added vocab (load directly). |
| `model/chat_template.jinja` | Generic Qwen3 ChatML template — **a DECOY**; `build_prompt` hand-writes the prompt and does NOT use this. Reference only. |

**External (not vendored, must obtain separately):** `descript-audio-codec` (`dac.nn.quantize` → `ResidualVectorQuantize`/`VectorQuantize` codebook lookup/projection/`from_codes`; `dac.nn.layers` → `Snake1d`, `WNConv1d`, `WNConvTranspose1d`) and `sglang.srt.layers.sampler` (`multinomial_with_seed` — reverse-engineer for seeded determinism: takes `log(probs)`, a clamped-nonneg seed, and the uncapped step counter). The per-layer Slow-AR forward op-order lives in the SGLang modeling code and should be confirmed there before finalizing fused-vs-split QKV and exact norm/RoPE placement.
