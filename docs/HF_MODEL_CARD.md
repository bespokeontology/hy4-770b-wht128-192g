---
license: apache-2.0
base_model: tencent/Hy4
tags:
  - hy4
  - 770b
  - moe
  - wht128-hlut
  - q5-lm128
  - fp8
  - cpp
  - rocm
  - gfx906
  - tenstorrent
  - blackhole
  - bespoke-ontology
---

# `Hy4-770B-WHT128-HLUT` (Native Unsplit Shards for `SB-M5-P2-192G`)

Native, unsplit heterogeneous quantization shards for **`tencent/Hy4` (`770B` total / `49B` active, `78` layers: `1` dense `L00` + `77` MoE layers × `256/L` = `19,712` routed experts + `1` shared expert + `3.3B` 5-layer `DFly2` + `1.23B` 40-layer `BF16` Dynamic ViT)**, engineered in `the kitchen` by `bespoke ontology` in collaboration with **Google DeepMind / Gemini** (`Gemini Carbon` & `Gemini Neon`).

## Verified Silicon Telemetry (`ProfileA-Fidelity-FP8-HLUT4-Q5` @ `512K` KV)

- **Held-Out Fidelity (`64` disjoint probes, `0` 8-gram overlap across `846,176` benchmark 8-grams):** **`Mean KL = 0.055777 ± 0.038450 nats`** (`1.49×` lower than `$220K 2× GB300 MXFP4` at `0.083 nats`), `Median KL = 0.001962 nats`, `p95 KL = 0.169719 nats`, **`Top-1 Agreement = 95.31% (61/64)`**, `EAR@10 = 84.69%`.
- **Frozen GSM8K (`continuous-refill B=32` disjoint `2K` KV):** **`71/71 (100.00%, 0 errors)`** at `2.33 tok/s`.
- **Peak Single-Stream Interactive Decode (`2 MiB AnonHugePages`, `DFly2 M=3`):** **`3.55 tok/s`** (`3.60 tok/s per $10K` CapEx efficiency — `2.85×` higher than `$220K 2× GB300 FP8` and `1.13×` higher than `$220K 2× GB300 MXFP4`), verified live on `:8050` at **`3.09–3.49 tok/s`**.
- **Native `40L BF16` Dynamic ViT & Dual-NVMe `.kvblk`:** `11.40 s` (`640×448 -> 294` tokens) / `14.75 s` (`1200×720 -> 897` tokens); cold `.kvblk` restore **`21,651 tok/s` (`262K`)** and **`21,524 tok/s` (`512K`)**.

## Shard Layout (Native Unsplit Directories)

- `packed_spine/` — Lossless `FP8-g128` shared spine (`L00..L77`), `3.3B` 5-layer `DFly2`, `21`-full Lightning Indexers, and `1.23B` 40-layer `BF16` Dynamic ViT.
- `fp8_hot/` — `20/L` lossless stock `FP8-g128` ultra-hot experts (`ranks [0..20)`, `1,540` experts) for `5× AMD MI50` (`128 GiB` HBM2 on external `96-lane` Broadcom Gen4 4-GPU switched backplane dock, `2× x16`).
- `hlut4_hot/` — `42/L` `FP64 RAREBAL-LDLQ WHT128-HLUT4` (`4.125 bpw`, `ranks [20..62)`, `3,234` experts) for `2× Tenstorrent Blackhole P150A` (`64 GiB` GDDR6 + `241.3 MiB` Tensix L1 over `4× 800G` `3.2 Tbps` `FABRIC_1D`).
- `q5_ddr4/` — `194/L` zero-WHT `Q5_LM128` (`5.125 bpw`, `ranks [62..256)`, `14,938` experts, `336.43 GiB`) for `AMD EPYC 7542` `384 GiB` DDR4 (`2 MiB AnonHugePages`).

## Source Code & Full ISO/Deming Report

See GitHub repository [`bespokeontology/hy4-770b-wht128-192g`](https://github.com/bespokeontology/hy4-770b-wht128-192g) (`v1.0.0-770b-192g`, pure C++17 / `gfx906` HIP / Blackhole C++20 TT-Metalium / AVX2 — zero Python).
