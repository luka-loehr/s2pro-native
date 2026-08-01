# s2pro-native — built inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (aarch64).
# See docs/SPARK.md. Finalized by the serve builder + integration.

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

all: $(BUILD)/s2pro-server $(BUILD)/s2p-test

$(BUILD)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/%.cu.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/fso/fso_wrap.o: src/fso/fso_wrap.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(FSO_INC) -c $< -o $@

$(BUILD)/s2pro-server: $(C_OBJS) $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(GENCODE) $^ $(FSO_OBJS) -o $@ $(LIBS)

# test binary: everything except main.c plus tests/test_forward.c
$(BUILD)/s2p-test: $(filter-out $(BUILD)/main.o,$(C_OBJS)) $(CU_OBJS) $(CPP_OBJS) tests/test_forward.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c tests/test_forward.c -o $(BUILD)/test_forward.o
	$(NVCC) $(GENCODE) $(filter-out $(BUILD)/main.o,$(C_OBJS)) $(CU_OBJS) $(CPP_OBJS) \
	    $(BUILD)/test_forward.o $(FSO_OBJS) -o $@ $(LIBS)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
