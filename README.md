# 🚀 IvanderLab Colab AI Suite
### High-Performance, Zero-Token Google Colab GPU Deployments for Generative AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%20Tesla%20T4-green.svg)](https://developer.nvidia.com/cuda-zone)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![Colab Tier](https://img.shields.io/badge/Colab-100%25%20Free%20Tier%20Compatible-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A unified collection of production-ready, zero-token Google Colab notebooks curated and engineered by **IvanderLab**. Designed specifically for creators, voice actors, and developers who want to run state-of-the-art Generative AI models—including **SDXL/Pony image generation, low-latency realtime voice conversion, emotional few-shot TTS, AI song covers, and local narrative LLMs**—on Google Colab's Free Tier (NVIDIA Tesla T4 GPU).

---

## ⚡ Quick Access Catalog

Click any **Open in Colab** badge to launch the environment immediately in your browser:

| Notebook | Domain | Core Engine / Model | Hardware Profile | Launch |
| :--- | :--- | :--- | :---: | :---: |
| **🎨 [SD-WebUI Forge (Pony XL)](#1--sd-webui-forge-pony-xl-edition)** | Image Generation | SD-WebUI Forge • AutismMix SDXL • 18 LoRAs | Tesla T4 (FP16) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivanderlab/IvanderLab-Colab-Suite/blob/main/Ivan_Forge_Pony_Masterpiece.ipynb) |
| **🎙️ [W-Okada Realtime Voice Changer](#2--w-okada-realtime-voice-changer-rvc-v1--v2)** | Realtime Audio | MMVCServerSIO • RVC v1/v2 • ContentVec | Tesla T4 (~67ms) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivanderlab/IvanderLab-Colab-Suite/blob/main/Colab_RealtimeVoiceChanger.ipynb) |
| **🗣️ [GPT-SoVITS Emotional TTS](#3--gpt-sovits-zero--few-shot-emotional-tts)** | Voice Synthesis | GPT-SoVITS • Zero-Shot Voice Cloning | Tesla T4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivanderlab/IvanderLab-Colab-Suite/blob/main/Colab_GPT_SoVITS_TTS.ipynb) |
| **🎵 [Applio RVC Voice Studio](#4--applio-rvc-voice-studio)** | Cover & Training | Applio RVC • Edge-TTS • Audio Separator | Tesla T4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivanderlab/IvanderLab-Colab-Suite/blob/main/Applio.ipynb) |
| **🧠 [KoboldCpp x L3.1 Stheno 8B](#5--koboldcpp-x-l31-stheno-8b-narrative-engine)** | Narrative LLM | KoboldCpp • L3.1-8B-Stheno-v3.2 (Q5_K_M) | Tesla T4 (Offload) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivanderlab/IvanderLab-Colab-Suite/blob/main/Stheno_KoboldCpp_Colab.ipynb) |

---

## 🎨 1. SD-WebUI Forge (Pony XL Edition)

> **File:** [`Ivan_Forge_Pony_Masterpiece.ipynb`](Ivan_Forge_Pony_Masterpiece.ipynb)  
> **Key Capabilities:** AutismMix SDXL, 18 Curated Anime LoRAs with Visual Previews, ADetailer Facial Mesh, Danbooru TagComplete, Infinite Image Browser, Stealth Anti-Ban Launch System.

### Overview
Stable Diffusion WebUI Forge is an optimized fork of AUTOMATIC1111 providing up to **2x faster generation and 50% less VRAM allocation**. This suite delivers an out-of-the-box anime and illustration environment tuned for **Pony Diffusion V6 XL** and **AutismMix SDXL**. It bypasses Colab's automated static code scanners through an encrypted repository unban layer.

### Step-by-Step Walkthrough
1. **CELL 1: Setup Environment & Stealth Unban**  
   Prepares Python dependencies, pulls the optimized Forge core from `camenduru/forge`, and installs essential UI extensions:
   * `sd-webui-infinite-image-browsing`
   * `adetailer` (automated face and hand restoration)
   * `a1111-sd-webui-tagcomplete` (Danbooru auto-tag completion)
2. **CELL 2: High-Speed Model Downloader**  
   Downloads checkpoint weights via multi-threaded `aria2c` at ~80–120 MB/s:
   * **Base Checkpoint:** `AutismMix SDXL (autismmixSDXL_autismmixConfetti.safetensors)`
   * **VAE:** Native SDXL VAE (`sdxl_vae.safetensors`)
   * **18 Anime Style & Character LoRAs:** Gacha styles, dynamic lighting, watercolor, cyber-aesthetic, and expression modifiers.
3. **CELL 3: Launch Stealth Forge WebUI**  
   Initializes Forge with optimal GPU flags (`--opt-sdp-attention --enable-insecure-extension-access`) and provisions a public Cloudflare tunnel.

### Recommended Generation Parameters
* **Sampler:** `Euler a` or `DPM++ 2M Karras`
* **Sampling Steps:** `28 – 35`
* **CFG Scale:** `5.5 – 7.0` (Do not exceed 8.0 on Pony models to prevent chromatic aberration)
* **Resolution:** `832 x 1216` (Portrait) or `1024 x 1024` (Square)
* **Mandatory Positive Prefix:** `score_9, score_8_up, score_7_up, source_anime, rating_safe, ...`
* **Mandatory Negative:** `score_6, score_5, score_4, simple background, bad anatomy, deformed limbs`

---

## 🎙️ 2. W-Okada Realtime Voice Changer (RVC v1 & v2)

> **File:** [`Colab_RealtimeVoiceChanger.ipynb`](Colab_RealtimeVoiceChanger.ipynb)  
> **Key Capabilities:** Native CUDA Inference, 67ms GPU Latency, TorchScript JIT Fix, Auto-Dimensional Calibration, Cloudflare HTTP/2 Protocol.

### Overview
A heavily patched deployment of **w-okada / tg-develop `MMVCServerSIO`** engineered to eliminate common tensor mismatch runtime errors on PyTorch 2.x and Google Colab Free Tier.

### Engineering Breakthroughs
* **TorchScript JIT Subclass Tensor Fix:** Solves the upstream exception where RVC v2 (768-dim) models inherit from RVC v1 (256-dim) base classes, preventing `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [1, 256] but got: [1, 768]`. Our configuration forces `"disableJit": 1` in native CUDA eager execution.
* **Auto-Calibration Pipeline:** Automatically inspects checkpoint weights and configures `embChannels` (256 vs 768), `embOutputLayer` (Layer 9 vs Layer 12), and sampling rate (32k/40k/48k).
* **HTTP/2 Cloudflare Tunneling:** Mitigates Indonesian & Southeast Asian transpacific ISP UDP packet-drop by enforcing `--protocol http2`.

### Telemetry & Benchmarks (499 Realtime Frames)
| Metric | Mean | Median (50th) | Min | 95th Percentile | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPU Inference (`PerfValue`)** | **`67.72 ms`** | `65.00 ms` | `59.00 ms` | `83.00 ms` | **99.4% Good** |
| **Audio Chunk Size (`PerfChunk`)**| `162.06 ms` | `168.00 ms` | `131.00 ms` | `168.00 ms` | Stable |
| **Network Roundtrip (`Ping`)** | `641.20 ms` | `641.00 ms` | `314.00 ms` | `1,181.00 ms` | Network Bound |
| **Total End-to-End Latency** | `907.40 ms` | `907.00 ms` | `545.00 ms` | `1,371.00 ms` | Playable |

> *Full raw dataset: [`benchmarks/performance_telemetry.csv`](benchmarks/performance_telemetry.csv).*

### Recommended WebUI Settings
* **GPU ID:** `cuda:0 (Tesla T4)`
* **Chunk Size:** `170.7 ms` – `341.3 ms` (absorbs ~500ms network jitter without robotic stutter)
* **Extra Processing Time:** `0.5 s` (4096 samples)
* **Pitch Detection (f0):** `rmvpe`
* **FAISS Index Ratio:** `0.6 – 0.7`
* **Input Gate:** `-55 dB`

---

## 🗣️ 3. GPT-SoVITS (Zero / Few-Shot Emotional TTS)

> **File:** [`Colab_GPT_SoVITS_TTS.ipynb`](Colab_GPT_SoVITS_TTS.ipynb)  
> **Key Capabilities:** 5-Second Voice Cloning, Multilingual Synthesis (Indonesian, English, Japanese, Chinese), Emotional Intonation Control.

### Overview
Unlike RVC which requires a live microphone input to modulate pitch, **GPT-SoVITS** is a state-of-the-art Text-to-Speech system. It clones any character or human voice from a single short audio sample (3 to 10 seconds) and synthesizes completely new spoken sentences with authentic cadence and emotion.

### Step-by-Step Walkthrough
1. **CELL 1: Setup Environment & Base Weights**  
   Clones the repository, installs FFmpeg and PyTorch audio dependencies, and automatically retrieves base pretrained Chinese/English/Japanese HuBERT models.
2. **CELL 2: Model & Sample Ingestion**  
   Downloads pre-indexed character voice packages directly into the cache directory.
3. **CELL 3: Launch TTS WebUI**  
   Generates a Gradio and Cloudflare live tunnel.
4. **Usage in WebUI:**  
   * Upload or select a 5-second clean reference audio clip.
   * Provide the transcript of what was said in the reference clip.
   * Type any target text in the prompt box and click **Synthesize Speech**.

---

## 🎵 4. Applio RVC Voice Studio

> **File:** [`Applio.ipynb`](Applio.ipynb)  
> **Key Capabilities:** Edge-TTS Script Voiceover, AI Song Cover Generation, UVR5 Vocal & Instrumental Demixing, Offline Audio Processing.

### Overview
**Applio** (by IAHispano) is an advanced modular RVC ecosystem optimized for batch processing, automated voice generation, and high-fidelity vocal track conversion.

### Core Features
* **AI Song Cover Pipeline:** Input any MP3/WAV song; Applio automatically splits vocals and instrumental stems using UVR5 algorithms, shifts pitch, replaces the singer's voice with your chosen RVC model, and remixes the final track.
* **Edge-TTS Narrator:** Convert lengthy markdown or article scripts into spoken natural narration using Microsoft Edge neural voices (e.g., `id-ID-GadisNeural`, `id-ID-ArdiNeural`), and instantly convert them into anime character voices in one click.

---

## 🧠 5. KoboldCpp x L3.1 Stheno 8B (Narrative Engine)

> **File:** [`Stheno_KoboldCpp_Colab.ipynb`](Stheno_KoboldCpp_Colab.ipynb)  
> **Key Capabilities:** Llama 3.1 8B Architecture, Sao10K Stheno v3.2 Checkpoint, KoboldCpp C++ Engine, Native RisuAI & SillyTavern Tunnel.

### Overview
Runs **L3.1-8B-Stheno-v3.2**—one of the highest-rated open-weights narrative, creative writing, and roleplay models—on Google Colab Free Tier using the hyper-optimized C++ inference engine **KoboldCpp**.

### Features & Setup
1. **Anti-Sleep Keep-Alive:** Cell 1 plays a silent looping HTML5 audio buffer to prevent Chrome and Edge tabs from being suspended by Google Colab while working in another window.
2. **KoboldCpp Server:** Loads the `Q5_K_M` GGUF quantization fully into Tesla T4 VRAM with GPU offload (`-ngl 99`), delivering inference speeds of **~45+ tokens per second**.
3. **Integration with Frontends:**  
   * Copy the generated `https://xxxx.trycloudflare.com` URL.
   * In **RisuAI** or **SillyTavern**, set API to **KoboldCpp / Text Generation WebUI** and paste the URL.
   * Enjoy full local context window processing without token limits or API subscriptions.

---

## 📁 Repository Structure

```
IvanderLab-Colab-Suite/
├── Ivan_Forge_Pony_Masterpiece.ipynb    # SDXL & Pony V6 Stable Diffusion Forge Studio
├── Colab_RealtimeVoiceChanger.ipynb     # W-Okada Realtime Voice Changer (Patched RVC)
├── Colab_GPT_SoVITS_TTS.ipynb           # GPT-SoVITS Zero-Shot Emotional TTS
├── Applio.ipynb                         # Applio Song Cover & Edge-TTS Suite
├── Stheno_KoboldCpp_Colab.ipynb         # KoboldCpp Llama 3.1 8B Stheno LLM Engine
├── benchmarks/
│   └── performance_telemetry.csv        # 499-frame raw latency & GPU telemetry log
├── LICENSE                              # MIT License
└── README.md                            # Suite documentation & walkthroughs
```

---

## 💡 Best Practices for Free-Tier Colab Users

1. **Avoid Multiple Concurrent Sessions:** Google Colab Free limits each Google account to one active GPU instance at a time. Always disconnect your runtime before opening another notebook.
2. **Keep Browser Active:** While running realtime audio or LLM generation, ensure the browser tab remains active or run the embedded silent audio keep-alive cell.
3. **Zero Token Philosophy:** All notebooks in this suite use zero-token public tunnels (`Cloudflare Tunnel` / `trycloudflare.com`). No account registrations, credit cards, or API secrets are required.

---

## 👨‍💻 Author & Maintainer

**Andhika Ivander Nathanael (Nat / Ivander)**  
* **Brand:** [IvanderLab](https://github.com/ivanderlab) (`@ivanderlab`)
* **Role:** Creative Developer & DKV / Interactive Media Designer
* **Contact:** `contact.ivanderlab@gmail.com`
* **Socials:** [GitHub](https://github.com/ivanderlab) • [Instagram](https://instagram.com/ivanderlab) • [LinkedIn](https://linkedin.com/in/ivanderlab) • [Behance](https://behance.net/ivanderlab)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
