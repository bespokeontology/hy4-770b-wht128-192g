#pragma once
// Pure-C++17 / rocBLAS Native 40-Layer Dynamic ViT Vision Encoder for tencent/Hy4 (770B)
// Loads packed_spine/vit.safetensors (2.30 GiB BF16) via zero-VRAM host mmap & streams per-layer
// onto MI50 rocBLAS on demand (0 MiB permanent VRAM overhead, ZERO Python).
// Geometry: 40 blocks (D=1536, H=16, Dh=96, F=6144, GELU) + Perceive 2x2 merger -> out_dim=6144 + image_newline [6144]

#include "../src/hy4_770b_config.hpp"
#include "../src/safetensors_mmap.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hy4_770b {

struct Hy4ImageRgb {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;  // [height * width * 3]
};

struct Hy4VitOutput {
    int grid_h = 0;
    int grid_w = 0;
    int merged_h = 0;
    int merged_w = 0;
    int num_tokens = 0;
    double encode_ms = 0.0;
    std::vector<uint16_t> embeddings_bf16;  // [num_tokens * kHiddenSize(6144)] BF16
};

bool decode_image_bytes(const uint8_t* data, size_t len, Hy4ImageRgb& out, std::string* err = nullptr);
bool load_image_from_url_or_path(const std::string& url_or_path, Hy4ImageRgb& out, std::string* err = nullptr);

class Hy4770bVitEncoder {
public:
    Hy4770bVitEncoder();
    ~Hy4770bVitEncoder();
    Hy4770bVitEncoder(const Hy4770bVitEncoder&) = delete;
    Hy4770bVitEncoder& operator=(const Hy4770bVitEncoder&) = delete;

    bool init(const std::string& vit_safetensors_path, int hip_dev = 0);
    bool ready() const noexcept;
    Hy4VitOutput encode_image(const Hy4ImageRgb& img) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hy4_770b
