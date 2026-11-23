# Pure-C++17 / gfx906 HIP / Blackhole C++20 TT-Metalium / EPYC AVX2 Build System
# Model: tencent/Hy4 (770B / A49B) on SB-M5-P2-192G (Zero Python anywhere)

CXX      ?= g++
HIPCC    ?= /opt/rocm/bin/hipcc
GCC_DIR  ?= /usr/lib/gcc/x86_64-linux-gnu/11

CXXFLAGS := -std=c++17 -O3 -march=native -fopenmp -pthread -Wall -Wextra -Werror -Wno-unused-parameter
AVXFLAGS := -std=c++17 -O3 -mavx2 -mfma -mf16c -msse4.2 -fopenmp -pthread -Wall -Wextra -Werror -Wno-unused-parameter
HIPFLAGS := --gcc-install-dir=$(GCC_DIR) -fPIC -std=c++17 -O3 --offload-arch=gfx906 -fno-gpu-rdc \
            -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-command-line-argument
ROCM_INC := -I/opt/rocm/include -Isrc -Imi50 -Ip150
ROCM_LIB := -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -lrocblas -lpcre2-8 -fopenmp -pthread -lz -lrt

BUILD := build/mi50
BIN   := bin

CORE_OBJS := $(BUILD)/hy4_770b_hlut_kernels.o \
             $(BUILD)/hy4_770b_core_kernels.o \
             $(BUILD)/hy4_770b_ddr4_avx2.o \
             $(BUILD)/hy4_770b_vit.o \
             $(BUILD)/hy4_tokenizer.o \
             $(BUILD)/hy4_770b_runtime.o

BINS := $(BIN)/hy4_770b_fe \
        $(BIN)/hy4_770b_gate_eval \
        $(BIN)/hy4_770b_stream_pack \
        $(BIN)/hy4_770b_pack_fp8_q5 \
        $(BIN)/hy4_770b_verify_kernels \
        $(BIN)/hy4_770b_sanitize \
        $(BIN)/hy4_770b_chart_gen \
        $(BIN)/hy4_770b_release_verifier \
        $(BIN)/hy4_770b_fetch_headers

.PHONY: all p150 audit-gfx906 verify-unit clean

all: $(BINS)

$(BUILD) $(BIN):
	@mkdir -p $(BUILD) $(BIN)

$(BUILD)/hy4_tokenizer.o: mi50/hy4_tokenizer.cpp mi50/hy4_tokenizer.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Imi50 -c $< -o $@

$(BUILD)/hy4_770b_hlut_kernels.o: mi50/hy4_770b_hlut_kernels.hip mi50/hy4_770b_hlut_kernels.hpp src/hy4_770b_config.hpp | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -Isrc -Imi50 -c $< -o $@

$(BUILD)/hy4_770b_core_kernels.o: mi50/hy4_770b_core_kernels.hip mi50/hy4_770b_core_kernels.hpp src/hy4_770b_config.hpp | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -Isrc -Imi50 -c $< -o $@

$(BUILD)/hy4_770b_ddr4_avx2.o: mi50/hy4_770b_ddr4_avx2.cpp mi50/hy4_770b_ddr4_avx2.hpp src/hy4_770b_config.hpp src/hy4_770b_hlut_pack.hpp | $(BUILD)
	$(CXX) $(AVXFLAGS) -Isrc -Imi50 -c $< -o $@

$(BUILD)/hy4_770b_vit.o: mi50/hy4_770b_vit.cpp mi50/hy4_770b_vit.hpp src/hy4_770b_config.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) -c $< -o $@

$(BUILD)/hy4_770b_runtime.o: mi50/hy4_770b_runtime.cpp mi50/hy4_770b_runtime.hpp mi50/hy4_770b_ddr4_avx2.hpp mi50/hy4_770b_nvme_kv_cache.hpp mi50/hy4_770b_vit.hpp mi50/hy4_770b_core_kernels.hpp mi50/hy4_770b_hlut_kernels.hpp src/hy4_770b_config.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) -c $< -o $@

$(BIN)/hy4_770b_fe: mi50/hy4_770b_fe.cpp $(CORE_OBJS) | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(CORE_OBJS) $(ROCM_LIB) -o $@

$(BIN)/hy4_770b_gate_eval: tools/hy4_770b_gate_eval.cpp $(CORE_OBJS) | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(CORE_OBJS) $(ROCM_LIB) -o $@

$(BIN)/hy4_770b_probe_diag: tools/hy4_770b_probe_diag.cpp $(CORE_OBJS) | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(CORE_OBJS) $(ROCM_LIB) -o $@

$(BIN)/hy4_770b_stream_pack: tools/hy4_770b_stream_pack.cpp $(BUILD)/hy4_770b_hlut_kernels.o $(BUILD)/hy4_770b_core_kernels.o $(BUILD)/hy4_tokenizer.o | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(BUILD)/hy4_770b_hlut_kernels.o $(BUILD)/hy4_770b_core_kernels.o $(BUILD)/hy4_tokenizer.o $(ROCM_LIB) -o $@

$(BIN)/hy4_770b_pack_fp8_q5: tools/hy4_770b_pack_fp8_q5.cpp src/hy4_770b_config.hpp src/hy4_770b_hlut_pack.hpp src/hy4_770b_range_stream.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/hy4_770b_verify_kernels: tools/hy4_770b_verify_kernels.cpp $(BUILD)/hy4_770b_hlut_kernels.o $(BUILD)/hy4_770b_core_kernels.o $(BUILD)/hy4_770b_ddr4_avx2.o | $(BIN)
	$(CXX) $(AVXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(BUILD)/hy4_770b_hlut_kernels.o $(BUILD)/hy4_770b_core_kernels.o $(BUILD)/hy4_770b_ddr4_avx2.o $(ROCM_LIB) -o $@

$(BIN)/hy4_770b_fetch_headers: tools/hy4_770b_fetch_headers.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/hy4_770b_sanitize: tools/hy4_770b_sanitize.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/hy4_770b_chart_gen: tools/hy4_770b_chart_gen.cpp mi50/hy4_json.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Imi50 $< -o $@

$(BIN)/hy4_770b_release_verifier: tools/hy4_770b_release_verifier.cpp src/hy4_770b_config.hpp src/hy4_770b_hlut_pack.hpp mi50/hy4_cycle_breaker.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc -Imi50 $< -o $@

p150:
	bash p150/build.sh

audit-gfx906: | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -Isrc -Imi50 --cuda-device-only -S mi50/hy4_770b_hlut_kernels.hip -o $(BUILD)/hy4_770b_hlut_kernels_gfx906.s
	$(HIPCC) $(HIPFLAGS) -Isrc -Imi50 --cuda-device-only -S mi50/hy4_770b_core_kernels.hip -o $(BUILD)/hy4_770b_core_kernels_gfx906.s
	@echo "=== gfx906 ISA Zero-Scratch Audit ==="
	@for f in $(BUILD)/hy4_770b_hlut_kernels_gfx906.s $(BUILD)/hy4_770b_core_kernels_gfx906.s; do \
	  tot=$$(grep -c "ScratchSize:" "$$f"); \
	  zer=$$(grep -c "ScratchSize: 0$$" "$$f"); \
	  trig=$$(grep -cE "v_(sin|cos)_f32" "$$f" || true); \
	  slc=$$(grep -E "(global|flat)_atomic_" "$$f" | grep -c "slc" || true); \
	  echo "  $$f: $$zer/$$tot ScratchSize=0 | trig=$$trig | slc_atomic=$$slc"; \
	  test "$$tot" -gt 0 && test "$$tot" -eq "$$zer" && test "$$trig" -eq 0 && test "$$slc" -eq 0; \
	done
	@echo "[AUDIT-GFX906] PASS: 100% Zero-Scratch, 0 Spills, 0 Trig, 0 SLC Atomics!"

verify-unit: $(BIN)/hy4_770b_release_verifier
	$(BIN)/hy4_770b_release_verifier --unit-only

clean:
	rm -rf $(BUILD) $(BINS)
