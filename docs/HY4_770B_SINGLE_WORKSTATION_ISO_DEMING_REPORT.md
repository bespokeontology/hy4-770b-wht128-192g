# ISO/Deming Technical Report: `tencent/Hy4` (770B Total / 49B Active) on a Single `<$10K` Workstation (`SB-M5-P2-192G`)

- **Document Control:** `HY4-770B-ISO-DEMING-FINAL-20261123-V1.0`
- **Standards Alignment:** `ISO 9001:2015` (Quality Management & Traceable Change Control), `ISO/IEC 25010` (Performance Efficiency, Functional Suitability, Reliability), `ISO/IEC 17025` (Metrology, Uncertainty Reporting & Reproducibility), W. Edwards Deming PDCA & Statistical Process Control (SPC)
- **Authors:** `bespoke ontology` (`the kitchen`) & Gemini (Google DeepMind — Gemini Carbon & Gemini Neon)
- **Target Model:** `tencent/Hy4` (`770B` total / `49B` active, `78` layers: `1` dense `L00` + `77` MoE layers × `256/L` = `19,712` routed experts + `1` shared expert `[6144, 2048]` + `3.3B` 5-layer `DFly2` + `1.23B` 40-layer `BF16` Dynamic ViT)
- **Immutable Release Freeze:** `FREEZE_HY4_770B_192G_20261122` (`chmod -R a-w`, FIPS 180-4 `SHA256SUMS_FREEZE.txt`)

---

## §1. Executive Summary & Headline Comparison vs. `$220K` Two-Station Baseline

This report documents the end-to-end pure-C++17 / `gfx906` HIP / Blackhole C++20 TT-Metalium / EPYC AVX2+FMA+F16C deployment of **`tencent/Hy4` (`770B` / `A49B`)** at full `512K` context length on a single workstation (`SB-M5-P2-192G`, `<$10,000` CapEx) with **zero Python** anywhere in the build, packing, runtime, verification, or chart pipeline.

Against the published `$220,000` two-station reference (`2× DGX Station GB300 + 2× RTX PRO 6000 + 400GbE`), our winning **`ProfileA-Fidelity-FP8-HLUT4-Q5`** residency achieves:
1. **`1.49×` lower held-out KL divergence (`0.055777 ± 0.038450 nats`)** than `$220K` `2× GB300 MXFP4` (`0.083 nats`), with **`95.31%` (`61/64`) Top-1 agreement** (clearing the `> 93.8%` threshold by `+1.51 pp`), `Median KL = 0.001962 nats`, `p95 KL = 0.169719 nats`, `Top-10 KL = 0.051602 nats`, `EAR@10 = 84.69%`, and `0` 8-gram overlap across `846,176` benchmark 8-grams.
2. **`3.55 tok/s` peak single-stream interactive decode** (`DFly2 M=3`, `2 MiB AnonHugePages`, `26` physical Zen 2 cores, `3.09–3.23 tok/s` verified live on `:8050` OpenAI/SSE) — delivering **`3.60 tok/s per $10K` hardware CapEx**, outperforming BOTH `$220K 2× GB300 stock FP8` (`1.26 tok/s per $10K`, **`2.85×` higher CapEx efficiency**) AND `$220K 2× GB300 MXFP4` (`3.18 tok/s per $10K`, **`1.13×` higher CapEx efficiency**).
3. **`100.00%` (`71/71`, `0` errors)** frozen GSM8K parity under continuous-refill `B=32` disjoint `2K` KV batching (`2.33 tok/s`), **`32/32` (`100%`)** `DFly2` bit-exact equivalence (`M=1 == M=6`), **`11.40 s`** on-demand `40L BF16` Dynamic ViT (`294` visual tokens, `0 B` permanent VRAM), **`21.7K / 21.5K tok/s`** cold Dual-NVMe `.kvblk` restore at `262K` / `512K`, **`57/57` zero-scratch `gfx906` ISA audit**, **`36/36` unit tests PASS**, and **`0 / 0` PCIe AER** across all `5/5` MI50s and `69/69` system PCIe endpoints.

---

## §2. Ground-Truth Hardware Topology & 4-Tier Memory Residency (`SB-M5-P2-192G`)

Every hardware identifier was audited directly from Linux sysfs, `lscpu`, ROCm HIP runtime telemetry, and Tenstorrent Blackhole mesh telemetry:

- **Host CPU & DDR4:** Single-socket, single-NUMA **`AMD EPYC 7542 32-Core Processor`** (`32` physical cores / `64` logical threads across `8` CCDs, `128 MiB` L3), `384 GiB` (`377 GiB` kernel-visible) 8-channel DDR4.
  - Because Linux `6.8.0-138-generic` disables `CONFIG_READ_ONLY_THP_FOR_FS` (`FileHugePages = 0 kB`), `mmap(MAP_SHARED)` incurs `88.2M` `4 KiB` PTEs that thrash Zen 2's L2 dTLB. Our `O_DIRECT` anonymous hugepage loader (`MAP_PRIVATE | MAP_ANONYMOUS` + `MADV_HUGEPAGE` after `POSIX_FADV_DONTNEED`) maps all `336.43 GiB` of `Q5_LM128` weights into **`336.57 GiB` live `2 MiB AnonHugePages`** (`172,326` 2M PTEs) in `105.4–106.2 s` (`3.40–3.43 GB/s`), leaving `20.9 GiB` available RAM and **`0 B` swap** (`pswpin = 0`, `pswpout = 0`).
- **GPU Compute Tier (`128 GiB` HBM2):** **`5× AMD Radeon Instinct MI50`** (`gfx906`, `3× 32 GiB` + `2× 16 GiB`, VRAM-sorted `R0=HIP0 [32G]`, `R1=HIP2 [32G]`, `R2=HIP3 [32G]`, `R3=HIP1 [16G]`, `R4=HIP4 [16G]`), attached via an external **`96-lane` Broadcom Gen4 4-GPU switched backplane dock (`2× x16 = 64 GB/s` host uplinks)** plus direct host PCIe slot.
- **ASIC Accelerator Tier (`64 GiB` GDDR6 + `330 MiB` Tensix L1):** **`2× Tenstorrent Blackhole P150A`** (`32 GiB` GDDR6/card) interconnected over a **`4× 800G` (`3.2 Tbps`) `FABRIC_1D`** mesh, pinned to dedicated CCD 6 (`cores 24-27`), serving over lock-free POSIX shared memory (`/hy4_770b_p150_ipc`) with `POSIX_FADV_DONTNEED` weight-upload guard.
- **Dual-NVMe `.kvblk` Backing Store:** Striped `O_DIRECT` persistence across two NVMe volumes (`50/50` layer + indexer striping) with hardware SSE4.2 CRC32C chunk verification and single-stripe streaming (`<= 13.2 GiB` peak host RAM at `512K`, preventing any page-cache eviction of DDR4 weights).

### Active Production Residency (`ProfileA-Fidelity-FP8-HLUT4-Q5` @ `512K` KV)

| Tier | Resident Assets (`78`-Layer `Hy4 770B`: `1` dense `L00` + `77` MoE layers × `256/L` = `19,712` routed experts) | Footprint | Headroom & Isolation |
| :--- | :--- | :--- | :--- |
| **Tier 0: Blackhole Tensix L1** | `13` ultra-hot `HLUT4` shards/card (`26` shards total across `2× P150A`) | `241.31 MiB` | `256 KiB/core` CB guard active |
| **Tier 1A: `5× MI50` HBM2** | Full FP8 spine (`2:2:2:1:1` TP5) + `3.3B` 5L `DFly2` + `20/L` stock `FP8-g128` experts (`[5,5,5,3,2]`, `1,540` total) + `512K` KV (`30.91 GiB`: `L00..L69` `8×64` FP8 + `L70..L77` lossless BF16) + `21` full indexers | `[30.67, 29.52, 29.52, 14.83, 12.12] GiB` (`116.66 / 128 GiB`) | **`3.86 GiB` free on `R4`** for on-demand `40L BF16` Dynamic ViT (`0 B` permanent VRAM) |
| **Tier 1B: `2× P150A` GDDR6** | `42/L` `WHT128-HLUT4` (`4.125 bpw`) experts (`ranks [20..62)`, `3,234` experts) | `29.31 GiB/card` (`58.62 GiB`) | `~2.69 GiB/card` headroom; pinned to CCD 6 (`cores 24-27`) |
| **Tier 2: `EPYC 7542` DDR4** | `194/L` zero-WHT `Q5_LM128` (`5.125 bpw`) cold experts (`ranks [62..256)`, `14,938` experts) | `336.43 GiB` (`336.57 GiB` live `2 MiB AnonHugePages`) | `20.9 GiB` available RAM, `0 B` swap; `26` isolated physical cores (`2..23, 28..31`) |

---

## §3. Quantization Mathematics, Streaming Packer & Zero-Overlap Calibration

1. **Multi-Precision Heterogeneous Quantization:**
   - **Stock `FP8-g128` (MI50 Tier 1A, `20/L`):** Lossless native `tencent/Hy4` `E4M3` block-128 weights (`37.76 MB/expert`) preserving maximum fidelity on the top-20 routed experts per layer.
   - **FP64 `RAREBAL-LDLQ WHT128-HLUT4` (Blackhole P150A Tier 1B, `42/L`):** `128`-point Walsh-Hadamard transform with deterministic sign flips + exact FP64 Lloyd-Max 16-level codebook (`20.22 dB` SQNR, residual `9.94e-13`), packed in `528 B` `32×32` tiles (`4.125 bpw`, `19.90 MB/expert`).
   - **Zero-WHT `Q5_LM128` (EPYC DDR4 Tier 2, `194/L`):** 5-bit symmetric Lloyd-Max magnitude-plus-sign block-128 format (`82 B/block`: `64 B` packed 4-bit magnitudes + `16 B` sign bits + `2 B` FP16 scale = `5.125 bpw`, `24.18 MB/expert`). Eliminates all runtime WHT128 overhead on CPU while delivering `cosine = 0.99999999` vs. exact FP32 unpack via overflow-proof AVX2 `vpshufb + vpmaddwd`.
   - **Hybrid `512K` KV Precision (`8×64` FP8 + `L70..L77` Lossless BF16):** Root-cause diagnostics on `Probe 51` revealed massive activation amplification across `L75 -> L76` (`RMS 57.5 -> 5,726.3`) that saturated coarse FP8 KV scales. Adopting `8×64` FP8 sub-blocks on `L00..L69` plus lossless `BF16` KV on `L70..L77` reduced `Probe 51` KL from `7.4419 nats` to `< 0.15 nats` and cut overall `p95 KL` by `2.04×` (`0.3460 -> 0.1697 nats`).
2. **Calibration Disjointness Receipt (`artifacts/cal_corpus_disjoint_receipt.json`):** `8,192` calibration tokens (`16 × 512`), audited against `846,176` benchmark 8-grams across held-out probes, GSM8K, MMLU-Pro, and Needles with **`0` overlapping 8-grams** (`CRC32C = 0x6792f545`).

---

## §4. Deming PDCA Engineering Audit Trail (Cycles 1–7)

| PDCA Cycle | Action & Root Cause | Measured Outcome |
| :--- | :--- | :--- |
| **Cycle 1 (`A3`)** | `113/L` HLUT3 VRAM + `143/L` HLUT4 DDR4 (`0.1748 nats`, `58/64`). Fixed `launch_bf16_gemv_bf16` (`M>1`) & `TP5` `kSharedInterSize`. | Eliminated multi-token spine bias; established need for `>= 4` bits on hot experts. |
| **Cycle 2 (`A4`)** | `84/L` HLUT4 VRAM + `172/L` HLUT4 DDR4 (`0.2174 nats`, `Median 0.0038`, `Probe 51 = 7.44`). Root-caused `L75->L76` massive-act KV overflow & `MADV_RANDOM` prefault stall. | Fixed `23.3 min -> 2.06 s` mmap load; isolated `L70..L77` BF16 KV requirement. |
| **Cycle 3 (`A5 v1`)** | `20/L` FP8 (MI50) + `42/L` HLUT4 (P150A) + `194/L` `Q5_LM128` (DDR4); BF16 KV `L74..L77`. | **`0.058144 nats`** (`1.43×` better than MXFP4), `60/64` (`93.75%`, 1 probe shy). |
| **Cycle 4 (`A5 v2`)** | Upgraded `L00..L69` KV to `8×64` FP8 sub-blocks + expanded lossless BF16 KV to `L70..L77`. | **HARD `GATE-2 v2` PASS:** **`0.055777 ± 0.038450 nats`** (`1.49×` lower than MXFP4), **`95.31%` (`61/64`) Top-1**, `Median 0.001962`, `p95 0.169719`. |
| **Cycle 5 (`DFly2` & `GATE-4`)** | Fixed 5 architectural `DFly2` bugs (`slot0` fusion, causal draft KV merge, autoregressive `hidden_correction`, multi-slot IndexCache, `all_h[accepted]` rollback) + hoisted `linear_gate` before `uv_bmm` + single-stripe `.kvblk` streaming (`<= 13.2 GiB` peak host RAM). | `32/32` (`100%`) `DFly2` bit-exact; `16K` acceptance **`48.6%` (`3.43 tok/step`, `CV=0.096%`)**; `40L ViT` `11.40 s`; cold `.kvblk` `21.7K / 21.5K tok/s` (`262K/512K`, `0/0` AER). |
| **Cycle 6 (2M THP & Cont. `B=32`)** | Replaced `4 KiB` `mmap` (`88.2M` PTEs) with `O_DIRECT` `2 MiB AnonHugePages` (`336.57 GiB`, `172K` PTEs); built disjoint `2K` KV + logical RoPE continuous-refill `B=32` evaluator; reverted `u8×i8` `Q5` experiment (`cos=0.99994` rejected by `GATE-1` floor) to spill-free `int16` `vpshufb+vpmaddwd` (`cos=0.99999999`). | **`71/71 (100.00%)`** frozen GSM8K at steady `cont_B32_tok_s = 2.33 tok/s`; `0 B` swap; `0/0` AER. |
| **Cycle 7 (Single-Stream Serving)** | Pinned `P150A` (`PID=6140`) to CCD 6 (`24-27`); isolated `26` physical DDR4 cores (`2..23, 28..31`); killed hidden 64-thread `#pragma omp` in `wht128_rows` & per-layer reductions; AVX2+F16C vectorized MLA online-softmax; removed `155` redundant pinned-H2D syncs; added dual-cache-line (`+0`/`+64` B) `Q5_LM128` prefetch + `<0.8%` Phase-2 balance; added configurable `DFly2 M=1..6`; fixed `kv_len == cached_toks.size()` on `DFly2` stop for `POST /v1/kv/save`. | Single-stream decode jumped to **`3.55 tok/s`** (`DFly2 M=3`, **`3.60 tok/s per $10K`**), `99.3%–100%` 3-tier overlap (`0.04 ms` outside DDR4), live `:8050` verified at **`3.09–3.23 tok/s`**. |

---

## §5. Verification Gate Ledger (`GATE-0` through `GATE-4`)

### `GATE-0` & `GATE-1` (Build, `gfx906` ISA Audit, Unit Tests, Kernel Math)
- **Compiler Hygiene:** `0` warnings under `-Wall -Wextra -Werror` across all 9 C++17/HIP/AVX2 binaries + C++20 TT-Metalium P150A worker; **`0` `.py` files**.
- **`gfx906` ISA Audit (`artifacts/hy4_770b_gfx906_isa_audit.json`):** `27/27` (`hy4_770b_hlut_kernels_gfx906.s`) + `30/30` (`hy4_770b_core_kernels_gfx906.s`) = **`57/57` `ScratchSize: 0`**, `0` SGPR spills, `0` VGPR spills, `0` device trig ops, `0` SLC system atomics.
- **Deterministic Unit Suite (`bin/hy4_770b_release_verifier`):** **`36/36 PASS`**.
- **`GATE-1` Kernel Equivalence (`bin/hy4_770b_verify_kernels`):** GPU HLUT2/3/4 `1.00000000`, Grouped HLUT4 `M=6` `1.00000000`, Grouped FP8 `M=6` `1.00000000`, Lossless BF16 MLA (`RMS=500`) `1.00000000`, CPU AVX2 HLUT4 `1.00000000`, CPU AVX2 `Q5_LM128` `0.99999999`.

### `GATE-2 v2` (`64`-Probe Disjoint Held-Out Fidelity @ `512K` KV)
- **`ProfileA-Fidelity-FP8-HLUT4-Q5` (`v2`):** `Mean KL = 0.055777 ± 0.038450 nats` (`1.49×` lower than `$220K MXFP4` at `0.083`), `Median KL = 0.001962 nats`, `p95 KL = 0.169719 nats`, `Top-10 KL = 0.051602 nats`, **`Top-1 = 95.31% (61/64)`**, `EAR@10 = 84.69%`, `Mean Logit Cos = 0.983285`.
- **`ProfileA-Fidelity-FP8-HLUT4-Q5` (`v1`, all-FP8 KV comparator):** `Mean KL = 0.058144 nats`, `Top-1 = 93.75% (60/64)`, `EAR@10 = 83.91%`.

### `GATE-3` (Frozen GSM8K Under Continuous-Refill `B=32` Disjoint `2K` KV)
- Verified **`71/71 (100.00%, 0 errors)`** frozen GSM8K completions at steady `cont_B32_tok_s = 2.33 tok/s` (`artifacts/gate3_profile_a5_512k.json`, gracefully drained on operator pivot to interactive serving priority with `RC=0` and `0/0` PCIe AER).

### `GATE-4` (`2K -> 512K` `n=3` SPC Scaling, `40L ViT`, Dual-NVMe `.kvblk`)
- **`DFly2` Exactness:** `32/32` (`100%`) bit-exact (`M=1 == M=6`); `64/64` confirmed at `M=2` and `M=6` on `math_gsm8k`.
- **Cold `4K` Prefill:** `4,096` tokens in `919.90 s` = `4.45 tok/s` (`9.49 tok/s` on `62`-token interactive prompt).
- **`40L BF16` Dynamic ViT (`640×448`):** `11,402.8 ms` (`11.40 s`), `294` visual tokens (`grid 28×40`), `rms = 0.2402`, `ok = 1`.
- **`2K -> 512K` `n=3` SPC Ladder (`artifacts/gate4_profile_a5_512k.json`):** `2K`: `1.26 ± 0.02 tok/s` (`acc=37.2%`, `CV(verify)=0.53%`); `16K`: `1.49 ± 0.05 tok/s` (`acc=48.6%`, `verify=2131.91 ± 2.05 ms`, `CV=0.096%`); `131K/262K/512K`: flat `draft = 155.3–157.0 ms` all the way to `512K` (`HELDOUT-SPEED = 1.40 tok/s`).
- **Cold Dual-NVMe `.kvblk` Restore (`O_DIRECT` + SSE4.2 CRC32C):** `262K`: save `19.12 s` / cold restore `12.11 s` (**`21,651 tok/s`**, `1.17 GB/s`, `ok=1`); `512K`: save `61.10 s` / cold restore `24.36 s` (**`21,524 tok/s`**, `1.16 GB/s`, `ok=1`).

---

## §6. Single-Stream Interactive Decode Profile (`2 MiB AnonHugePages`, `M = 1..6`)

Measured on `ProfileA-Fidelity-FP8-HLUT4-Q5` (`26` isolated physical Zen 2 cores `2..23, 28..31`, Blackhole `P150A` pinned to CCD 6 `24-27`, `OMP_WAIT_POLICY=PASSIVE`):

| Mode | `code_cpp_hip` (`tok/s`) | `math_gsm8k` (`tok/s`) | `draft` / `verify` | `attn` / L | `moe_wall` / L (`MI50` / `P150` / `DDR4`) | `DDR4` BW | 3-Tier Overlap |
|---|---:|---:|---:|---:|---|---:|---:|
| **`M=1`** | `2.41` | `2.31` | `0.0` / `415.7 ms` | `2.18 ms` | `2.79 ms` (`1.27` / `1.00` / `2.68 ms`, `3.9 exp/L`) | `35.3 GB/s` | `96.1%` hidden |
| **`DFly2 M=2`** | `3.06` | `3.18` | `27.1` / `583.0 ms` | `2.44 ms` | `4.58 ms` (`1.60` / `1.39` / `4.53 ms`, `7.5 exp/L`) | `39.9 GB/s` | `98.9%` hidden |
| **`DFly2 M=3` (Winner)** | **`3.55`** | **`3.10`** | **`41.4` / `698.8 ms`** | **`2.59 ms`** | **`5.91 ms` (`2.02` / `1.76` / `5.87 ms`, `9.9 exp/L`)** | **`40.7 GB/s`** | **`99.3%` hidden** |
| **`DFly2 M=4`** | `3.29` | `3.14` | `56.5` / `876.8 ms` | `2.80 ms` | `7.61 ms` (`2.13` / `1.91` / `7.59 ms`, `13.0 exp/L`) | `41.5 GB/s` | `99.7%` hidden |
| **`DFly2 M=6`** | `2.79` | `3.43` | `87.3` / `1136.6 ms` | `3.46 ms` | `10.69 ms` (`2.36` / `2.36` / `10.69 ms`, `18.4 exp/L`) | `41.7 GB/s` | `100.0%` hidden |

---

## §7. `GATE-5` Live `:8050` / `:3081` Campaign, Cold `.kvblk` Restore & `40L BF16` ViT Read-Back

- **Live Endpoint:** `bin/hy4_770b_fe` serving `http://127.0.0.1:8050/` (`Hy4-770B-WHT128-192G`, `ProfileA-Fidelity-FP8-HLUT4-Q5`, `512K` KV, `DFly2 M=3`, `40L BF16` ViT armed on `R4`), registered in the live `:3081` DeepSeek Harness (`settings.yaml`, preserving all existing lanes untouched).
- **Live Interactive Telemetry (`:8050` `/v1/chat/completions`, `DFly2 M=3`):**
  - Cold Turn 1 (`55` prompt toks, `0` cached): **`3.09 tok/s`**, `TTFT = 7,527.9 ms` (`7.31 tok/s` prefill), `56.2%` acc (`2.11 tok/step`), `52.4%` accelerator hit rate (`L1=4.3%`, `MI50=29.8%`, `P150=18.3%`, `DDR4=47.6%`).
  - Warm Prefix Turn 2 (`52/87` cached, `35` new toks): **`3.23 tok/s`**, `TTFT = 4,340.4 ms` (`8.06 tok/s` incremental prefill), `65.4%` acc (`2.29 tok/step`), `52.1%` accelerator hit rate.
- **Graceful Teardown, Cold Dual-NVMe `.kvblk` Cycle, `5/5` Recall & `40L BF16` ViT Chart Read-Back (PASS):**
  - **Graceful Shutdown:** `POST /v1/shutdown` cleanly drained all active clients and executed `hipDeviceSynchronize()` across all `5/5` MI50s (`RC=0`, `0/0` PCIe AER) while keeping the warm Blackhole P150A daemon resident (`--keep-p150`), followed by a clean `213.9 s` cold restart (`336.57 GiB` `2 MiB AnonHugePages` loaded in `104.59 s` @ `3.45 GB/s`).
  - **Dual-NVMe `.kvblk` Save & Cold Restore:** Turn 1 anchored 5 exact campaign values (`164` tokens); `POST /v1/kv/save` persisted the live 5-GPU KV + BF16 tail sidecar in **`17.03 ms`** (`prefix_hash = 16106461945071048763`); `POST /v1/clear_all_caches` wiped VRAM KV; `POST /v1/kv/restore` (`evict_page_cache_first = true`) cold-restored `8.86 MB` (`164` tokens) via `O_DIRECT` + SSE4.2 CRC32C in **`9.43 ms`** (`17,397 tok/s`, `disk = 4.06 ms`, `crc32c = 0.00019 ms`, `h2d = 5.18 ms`).
  - **`5/5` Exact Recall on Restored KV:** Turn 2 hit **`cached_prefix_tokens = 164` (`100%`)**, decoded at **`3.49 tok/s`** (`DFly2 M=3`, **`98.1%` acceptance**, **`2.96 tok/step`**), and recalled **`5/5`** anchored values verbatim (`KBL-770B-NOV23`, `0.055777`, `19712`, `336.57`, `3.55`).
  - **Native `40L BF16` Dynamic ViT Live PNG Read-Back:** Turn 3 streamed the rendered `1200×720` `artifacts/hy4_770b_single_station_telemetry.png` through `/v1/chat/completions` (`image_url` base64 PNG); on-demand `40L BF16` ViT on `R4` (`HIP4`, `3.86 GiB` headroom) encoded `1200×720 -> grid 46×76 -> merged 23×38` (**`897` visual tokens**) in **`14.75 s`** (`0 B` permanent VRAM) and decoded multimodal read-back at **`2.64 tok/s`** (`76.2%` DFly2 acceptance, identifying `Profile A5 v2 BF16` and `Profile A5 v1 FP8` bars from raw pixels).
  - **Sanitized Session Archive:** Exported as `artifacts/dsh-session-hy4-770b-sanitized.zip` (`zip -X`, `0` PII hits).

---

## §8. Hardware Safety, PCIe AER Telemetry & Zero-PII Compliance

- **Strict Unattended Silicon Safety:** Zero `kill -9`, `pkill -9`, `rocm-smi --gpureset`, or `tt-smi -r` invocations across the entire campaign. Every process exit (`SIGTERM` / `POST /v1/shutdown`) executes `hipDeviceSynchronize()` across all `5/5` MI50 devices before freeing VRAM or closing `/hy4_770b_p150_ipc`.
- **PCIe AER Integrity:** `0` correctable, `0` nonfatal, `0` fatal across all `5/5` MI50 GPUs (`0000:05:00.0`, `0000:08:00.0`, `0000:85:00.0`, `0000:88:00.0`, `0000:c7:00.0`) and `0/0` nonfatal/fatal across `69/69` AER-capable endpoints.
- **Zero-PII / Zero-Secret Hygiene:** Audited via pure-C++17 `bin/hy4_770b_sanitize` (8/8 checks, `0` hits) and `bin/hy4_770b_release_verifier --verify-all` (`failures=0`). Legacy freezes `FREEZE_WAVE10_192G_20261121`, `FREEZE_HY4_HLUT4_192G_20261122`, and `FREEZE_HY4_770B_CHECKPOINT_GATE2_20261123` remain strictly read-only and untouched.

---

## §9. Reproducibility, Freeze Manifest & Attribution

- **Immutable Release Freeze:** `FREEZE_HY4_770B_192G_20261122` (`chmod -R a-w`, FIPS 180-4 `SHA256SUMS_FREEZE.txt` verified).
- **Repositories:** GitHub `bespokeontology/hy4-770b-wht128-192g` (`v1.0.0-770b-192g`); Hugging Face `kapitoshkina-ai/Hy4-770B-WHT128-HLUT` (`base_model: tencent/Hy4`).
- **Attribution:** Engineered in `the kitchen` by `bespoke ontology`, co-authored with **Gemini (Google DeepMind — Gemini Carbon & Gemini Neon)**. Licensed under Apache 2.0.
