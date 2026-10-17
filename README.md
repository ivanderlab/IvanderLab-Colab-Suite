# 🎙️ W-Okada Realtime Voice Changer — Google Colab GPU Suite
### High-Performance, Zero-Token Deployment with Native CUDA Acceleration & Architecture Patch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivanderlab/RVC-Realtime-Voice-Changer-Colab/blob/main/Colab_RealtimeVoiceChanger.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%20Tesla%20T4-green.svg)](https://developer.nvidia.com/cuda-zone)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model Support](https://img.shields.io/badge/RVC-v1%20%26%20v2%20(768--dim)-purple.svg)](#architecture-patches)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A robust, enterprise-grade Google Colab deployment suite for **w-okada / tg-develop Realtime Voice Changer (`MMVCServerSIO`)**. Engineered for instantaneous, low-latency voice conversion on Colab Free Tier (Tesla T4) with pre-calibrated RVC v1 & v2 pipelines, automated model ingestion, zero-token Cloudflare HTTP/2 tunnels, and comprehensive fixes for TorchScript JIT runtime anomalies.

---

## 🌟 Key Features

* **⚡ Zero-Token Deployment**: Instant public HTTPS tunnel via Cloudflare Tunnel (`trycloudflare.com`) without requiring account registration or credit cards. Optional Ngrok integration with ultra-low latency Asia-Pacific (`ap`) routing.
* **🧠 Deep Architecture Fixes**: Resolves the notorious `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [1, 256] but got: [1, 768]` bug present in standard W-Okada Colab scripts.
* **🚀 Native CUDA FP16 Inference**: Automated hardware initialization locking to NVIDIA Tesla T4 (`cuda:0`), achieving an average inference time of **~67ms**.
* **📦 Auto-Calibration Pipeline**: Inspects raw PyTorch checkpoint state dictionaries (`cpt["weight"]`) to dynamically calibrate embedding dimensions (256-dim vs 768-dim), embedder output layers (Layer 9 vs Layer 12), and sampling rates (32k / 40k / 48k).
* **🔧 Ubuntu 24.04 (Noble) Runtime Fix**: Automatically patches C++ standard library (`libstdc++.so.6`) symlinks to prevent dynamic linker crashes with `sounddevice` and `libportaudio2`.

---

## 📊 Telemetry & Performance Benchmarks

Real-world streaming telemetry captured across **499 consecutive audio conversion frames** via Cloudflare Tunnel on an active client stream:

| Metric | Mean | 50th Percentile (Median) | Min | 95th Percentile | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPU Inference (`PerfValue`)** | **`67.72 ms`** | `65.00 ms` | `59.00 ms` | `83.00 ms` | **99.4% Good** |
| **Audio Chunk Size (`PerfChunk`)**| `162.06 ms` | `168.00 ms` | `131.00 ms` | `168.00 ms` | Stable |
| **Network Roundtrip (`Ping`)** | `641.20 ms` | `641.00 ms` | `314.00 ms` | `1,181.00 ms` | Network Bound |
| **End-to-End Latency** | `907.40 ms` | `907.00 ms` | `545.00 ms` | `1,371.00 ms` | Playable |

> *Full raw telemetry dataset is available in [`benchmarks/performance_telemetry.csv`](benchmarks/performance_telemetry.csv).*

### 📈 Latency Breakdown
```
[Client Mic] ──(128ms - 341ms Chunk)──> [Cloudflare HTTP/2] ──(Packet Transit)──> [Colab Backend]
                                                                                       │
                                                                                 [Tesla T4 GPU]
                                                                              (Inference: ~67ms)
                                                                                       │
[Speaker Output] <──(Jitter Buffer)── [Cloudflare HTTP/2] <──(Converted Audio)─────────┘
```

---

## 🛠️ Deep Engineering Patches

### 1. The TorchScript JIT Subclass Tensor Mismatch Bug
#### **Problem:**
In upstream `tg-develop/voice-changer`, RVC v2 models (`SynthesizerTrnMs768NSFsid`) inherit directly from RVC v1's `SynthesizerTrnMs256NSFsid` without redefining the `@torch.jit.export def infer(...)` method signature.
When the server defaults to CPU or attempts TorchScript compilation (`torch.jit.optimize_for_inference(torch.jit.script(model))`), TorchScript compiles the computational graph based on the base class's type annotations (`self.enc_p` initialized with 256 input channels). When fed 768-dimensional ContentVec feature vectors, the C++ TorchScript executor panics:
```
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [1, 256] but got: [1, 768].
```

#### **Solution:**
1. Our configuration layer injects `"disableJit": 1` directly into `/content/ML_Program/stored_setting.json`.
2. PyTorch executes the forward pass in **native CUDA eager mode** on the GPU, completely bypassing the flawed TorchScript inheritance graph.
3. Model inference runs with full tensor dimensions intact (`(192, 768)` linear projections) at sub-70ms speeds without crashes.

---

### 2. Multi-Layer Feature Extractor Calibration
Standard RVC models depend on HuBERT / ContentVec embeddings with varying output layer configurations:
* **RVC v1 (256-dim)**: Requires Layer 9 features followed by projection (`embOutputLayer: 9`, `useFinalProj: True`).
* **RVC v2 (768-dim)**: Requires raw 12th-layer transformer hidden states (`embOutputLayer: 12`, `useFinalProj: False`).

Our script inspects the model checkpoint headers directly and generates a strict, complete `params.json` for each slot, preventing feature distortion and dimension collapse.

---

### 3. Indonesian & Remote ISP Network Jitter Mitigation
When streaming audio over long distances (e.g., from Southeast Asia to US/EU datacenters), UDP-based tunnels (such as standard Cloudflare QUIC) frequently experience packet-shaping and aggressive packet loss by local ISPs (Telkomsel, Indihome), causing **500ms retransmit gaps** ("patah-patah").
* By forcing `--protocol http2` in the Cloudflare daemon, communication switches to multiplexed TCP with zero dropped frames.
* Setting client **Chunk Size to `341.3ms`** creates an optimal jitter buffer, preventing client audio buffer underruns.

---

## 📋 Recommended WebUI Settings Matrix

To achieve optimal voice clarity and seamless real-time conversation:

| Setting Parameter | Recommended Value | Technical Rationale |
| :--- | :---: | :--- |
| **GPU Selection** | `cuda:0 (Tesla T4)` | Locks hardware to NVIDIA GPU Tensor cores. |
| **Chunk Size** | **`170.7ms` – `341.3ms`** | Prevents buffer underruns over transpacific connections. |
| **Extra Processing Time** | **`0.5 s` (4096)** | Ensures seamless SOLA crossfade transitions between chunks. |
| **Input Sensitivity** | **`-55 dB` to `-60 dB`** | Eliminates ambient mic hiss while retaining gentle vocal attacks. |
| **Pitch Detector (f0)** | **`rmvpe`** | Deep neural network pitch tracking; immune to octave jump errors. |
| **Index Ratio** | **`0.6` – `0.7`** | Blends trained FAISS feature retrieval with HuBERT latent features. |
| **Echo Cancellation** | **OFF** | Prevents WebRTC acoustic echo filters from clipping high frequencies. |
| **Noise Suppression** | **OFF** | Preserves vocal breathiness and character timbres. |

---

## 🚀 Quickstart Guide

### 1. Launch in Google Colab
Click the badge above or navigate to [`Colab_RealtimeVoiceChanger.ipynb`](Colab_RealtimeVoiceChanger.ipynb) in Google Colab.

### 2. Execute Cells in Sequence
1. **CELL 1: Setup Environment & Engine**
   * Prepares Python environment, downloads prebuilt CUDA binaries, installs `aria2c`, and patches Ubuntu 24.04 libraries.
2. **CELL 1.5: Auto-Install Voice Models**
   * Downloads and calibrates voice slots at datacenter speeds:
     * **Slot 0**: Mythia Batford (RVC v2, 32kHz, Pitch `+12`)
     * **Slot 1**: Vestia Zeta (RVC v2, 40kHz, Pitch `+12`)
3. **CELL 2: Launch WebUI & Tunnel**
   * Calibrates GPU device profiles, disables TorchScript JIT, initiates Cloudflare HTTP/2 tunnel, and outputs the public URL.

### 3. Connect & Converse
1. Open the generated `https://xxxx.trycloudflare.com` URL in your browser.
2. Select **Slot 0** or **Slot 1**.
3. Toggle audio input mode to **Client** and speak!

---

## 📁 Repository Structure

```
├── Colab_RealtimeVoiceChanger.ipynb   # Main production-ready Jupyter Notebook
├── benchmarks/
│   └── performance_telemetry.csv      # 499-frame raw latency & GPU telemetry log
├── LICENSE                            # MIT Open Source License
└── README.md                          # Architecture documentation & benchmarks
```

---

## 👨‍💻 Author

**Andhika Ivander Nathanael (Nat)**  
* Web Technologies Specialist & DKV / Interactive Media Designer
* GitHub: [@ivanderlab](https://github.com/ivanderlab)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
