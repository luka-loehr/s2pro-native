# Building & running on NVIDIA DGX Spark (GB10, sm_121)

Target hardware: NVIDIA GB10 (DGX Spark) — compute capability 12.1
(`sm_121`), 48 SMs, unified LPDDR5X memory (~273 GB/s theoretical, ~220 GB/s
sustained in GEMV-shaped reads; this number sets the decode floor, see the
performance section of the README).

The host needs only Docker with the NVIDIA runtime. All builds run inside
`nvidia/cuda:13.0.3-devel-ubuntu24.04` — no toolkit, Python, or PyTorch on
the host.

## 1. fish-scales-ops objects (FP8 path)

```sh
docker run --rm -v "$PWD":/work -w /work nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    bash -c "ARCH=121a FSO_DIR=/work/3rdparty/fish-scales-ops FSO_OUT=/work/3rdparty/build scripts/build_fso.sh"
```

`dispatch.cuh` in fish-scales-ops handles `arch == 120 || arch == 121`
explicitly; the GEMM path is verified on sm_121 (cos 0.9993 vs a BF16 cuBLAS
reference). The FSO *attention* kernels are sm_120a-only and are not used.

## 2. Checkpoint

```sh
scripts/fetch_model.sh model          # safetensors + tokenizer + config from HF
```

The DAC codec ships as `codec.pth` (PyTorch). The native loader reads a raw
`codec.bin`/`codec.idx` pair instead (format documented in
`src/dac/dac_weights.c`). Convert once with any torch-capable container:

```sh
python3 tools/convert_codec_full.py --model /path/with/codec.pth --out codec-full
```

The full artifact (~1.6 GB FP32) includes the encoder conv stack and
quantizer pre-module, which activates voice-cloning encode
(`s2p_dac_encode`); a decode-only artifact loads fine but disables encode.
For a cloning run: `S2P_TEST_REF=<44.1k mono s16 wav>` +
`S2P_TEST_REF_TEXT=<transcript>` on `s2p-test`.

## 3. Build

```sh
docker run --rm -v "$PWD":/work -w /work \
    -e FSO_DIR=/work/3rdparty/fish-scales-ops \
    nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    make -j8 FSO_OBJS="/work/3rdparty/build/runner_121a.o /work/3rdparty/build/quant_121a.o"
```

Produces `build/s2pro-server` and `build/s2p-test`.

## 4. Smoke test

```sh
docker run --rm --gpus all -v "$PWD":/work -w /work \
    -v "$PWD/model":/model:ro -v /path/to/codec:/codec:ro \
    nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    ./build/s2p-test /model /codec
```

Prints per-stage timings, the first frames' codes, and writes
`/tmp/s2p_smoke.wav`. `S2P_INT8=1` switches the GEMMs to the per-channel
weight-only INT8 path (parity-proven, halves decode time and weight RAM);
`S2P_FP8=1` selects the fish-scales-ops FP8 path (parity-FAILED, kept for
measurement only).

## 5. Server

```sh
./build/s2pro-server --model-dir /model --codec-dir /codec --port 8010
curl -s localhost:8010/healthz
curl -s -X POST localhost:8010/v1/tts -d '{"text":"Hello from the Spark.","format":"wav"}' -o out.wav
```

## Proven nvcc flags (sm_121a)

```
-gencode=arch=compute_121a,code=sm_121a -O3 -std=c++17
--expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-Wno-psabi
--diag-suppress=20012,20013,20014,177,20050
-U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
-DENABLE_BF16 -DENABLE_FP8
```

Link: `-lcublas -lcuda -lnvrtc -lm -lpthread`.

## FSO integration notes

The shim (`src/fso/fso_wrap.cpp`) uses exactly:

```c++
namespace blockscale_gemm { namespace detail {
void fp8bs_quantize_1x128_packed(__nv_fp8_e4m3*, int32_t*, __nv_bfloat16 const*, int M, int K, cudaStream_t, bool use_ue8m0);
void fp8bs_quantize_128x128(__nv_fp8_e4m3*, float*, __nv_bfloat16 const*, int N, int K, cudaStream_t);
void repack_ue8m0_scales_sfb_for_sm120(int32_t*, float const*, int N_pad, int N_blocks_in, int K_blocks_per_row, cudaStream_t);
}}
using RunnerQ = tensorrt_llm::kernels::blockscale_gemm::
    CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
```

Weights are quantized once at load (`128x128` block scale + SFB repack),
activations per call (`1x128`, UE8M0). Constraint: the activation-scale
repack needs `K % 512 == 0` — all S2-Pro GEMM widths qualify (2560, 4096,
6144, 9728); violations fall back to BF16 with a logged warning.
