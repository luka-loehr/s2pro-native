# s2pro-native — built inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (aarch64).
# See docs/SPARK.md. Finalized by the serve builder + integration.
#
# Targets:
#   all / nvcc-build   server + smoke-test binary (link needs the prebuilt
#                      fish-scales-ops objects, mounted at /fso on the box)
#   syntax             compile-only pass over every TU (no link, no GPU)
#   selftest           build + run the host-side selftests (json, tokenizer)
#   clean

ARCH        ?= 121a
BUILD       ?= build
FSO_DIR     ?= /fso/fish-scales-ops
FSO_OBJS    ?= /fso/build/runner_$(ARCH).o /fso/build/quant_$(ARCH).o

NVCC        ?= nvcc
CC          ?= gcc

GENCODE     := -gencode=arch=compute_$(ARCH),code=sm_$(ARCH)
NVCCFLAGS   := $(GENCODE) -O3 -std=c++17 --expt-relaxed-constexpr \
               --expt-extended-lambda -Xcompiler=-Wno-psabi \
               --diag-suppress=20012,20013,20014,177,20050 \
               -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
               -DENABLE_BF16 -DENABLE_FP8 -Iinclude
CFLAGS      := -O2 -std=c11 -Wall -Wextra -Iinclude \
               -I/usr/local/cuda/include -D_GNU_SOURCE
FSO_INC     := -I$(FSO_DIR)/csrc/gemm/include -I$(FSO_DIR)/csrc/common/compat/include \
               -I$(FSO_DIR)/3rdparty/cutlass/include -I$(FSO_DIR)/3rdparty/cutlass/tools/util/include
LIBS        := -lcublas -lcuda -lnvrtc -lm -lpthread

C_SRCS      := $(shell find src -name '*.c')
CU_SRCS     := $(shell find src -name '*.cu')
CPP_SRCS    := src/fso/fso_wrap.cpp
C_OBJS      := $(patsubst src/%.c,$(BUILD)/%.o,$(C_SRCS))
CU_OBJS     := $(patsubst src/%.cu,$(BUILD)/%.cu.o,$(CU_SRCS))
CPP_OBJS    := $(BUILD)/fso/fso_wrap.o

# Everything except the server entry point, for test/selftest links.
LIB_OBJS    := $(filter-out $(BUILD)/main.o,$(C_OBJS)) $(CU_OBJS) $(CPP_OBJS)

all: $(BUILD)/s2pro-server $(BUILD)/s2p-test

# docs/SPARK.md invokes `make -j nvcc-build`.
nvcc-build: all

$(BUILD)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/tests/%.o: tests/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/%.cu.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/fso/fso_wrap.o: src/fso/fso_wrap.cpp src/fso/fso.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(FSO_INC) -Isrc/fso -c $< -o $@

$(BUILD)/s2pro-server: $(C_OBJS) $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(GENCODE) $^ $(FSO_OBJS) -o $@ $(LIBS)

$(BUILD)/s2p-test: $(LIB_OBJS) $(BUILD)/tests/test_forward.o
	$(NVCC) $(GENCODE) $^ $(FSO_OBJS) -o $@ $(LIBS)

$(BUILD)/s2p-parity: $(LIB_OBJS) $(BUILD)/tests/parity.o
	$(NVCC) $(GENCODE) $^ $(FSO_OBJS) -o $@ $(LIBS)

parity: $(BUILD)/s2p-parity

# ---- host-side selftests (json parser, tokenizer + prompt builder) --------

$(BUILD)/selftest_json: $(LIB_OBJS) $(BUILD)/tests/selftest_json.o
	$(NVCC) $(GENCODE) $^ $(FSO_OBJS) -o $@ $(LIBS)

$(BUILD)/selftest_tok: $(LIB_OBJS) $(BUILD)/tests/selftest_tok.o
	$(NVCC) $(GENCODE) $^ $(FSO_OBJS) -o $@ $(LIBS)

# selftest_tok needs a tokenizer.json; the repo model/ dir carries one.
SELFTEST_MODEL_DIR ?= model

selftest: $(BUILD)/selftest_json $(BUILD)/selftest_tok
	$(BUILD)/selftest_json
	$(BUILD)/selftest_tok $(SELFTEST_MODEL_DIR)

# ---- compile-only syntax pass (no link; .cu/.cpp still need nvcc+device) --

SYNTAX_C  := $(C_SRCS) $(wildcard tests/*.c)
SYNTAX_CU := $(patsubst src/%.cu,$(BUILD)/syntax/%.cu.o,$(CU_SRCS))

syntax: $(SYNTAX_CU) $(BUILD)/syntax/fso_wrap.o
	@for f in $(SYNTAX_C); do \
	    echo "$(CC) -fsyntax-only $$f"; \
	    $(CC) $(CFLAGS) -fsyntax-only $$f || exit 1; \
	done
	@echo "syntax: OK"

$(BUILD)/syntax/%.cu.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/syntax/fso_wrap.o: src/fso/fso_wrap.cpp src/fso/fso.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(FSO_INC) -Isrc/fso -c $< -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all nvcc-build syntax selftest parity clean
