# MorphServe

**MorphServe** is a dynamic, workload-aware serving framework for Large Language Models (LLMs) that enables elastic adaptation of both transformer layer precision and KV cache capacity at runtime. It achieves this by introducing:

- **Importance-guided Layer Swapping**: selectively morphing full-precision layers to quantized variants on-the-fly.
- **Pressure-aware KV Cache Resizing**: expanding and releasing KV blocks in response to memory pressure.
- Full compatibility with existing efficient serving techniques (e.g., FlashAttention, PagedAttention, dynamic batching).

This repository provides the code, configuration, and artifacts for reproducing paper results. 

## 🔧 Features

- **Runtime-adaptive mixed-precision execution** (FP16, W8, W4).
- **Token-level in-place layer swapping** with `cudaMemcpyAsync`.
- **Elastic PagedAttention extension** for dynamic KV cache resizing.
- **Minimal integration overhead** into SwiftLLM / vLLM-style engines.
- **Configurable performance vs. accuracy trade-offs** (runtime modes).


## 🚀 Getting Started

### Requirements

- Python 3.10+
- CUDA 12.7+
- PyTorch 2.0+
- NVIDIA L4 (24GB) / A100 (80GB) recommended

Install AWQ first following the instructions: https://github.com/mit-han-lab/llm-awq
Create new env:
```
conda create -n morphserve python=3.10 -y
conda activate morphserve
```
install MorphServe
```
cd MorphServe
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install -e .
```

## 📊 Evaluation Setup

### Models

| Model Name       | Size | Attention | Hardware           |
|------------------|------|-----------|--------------------|
| Vicuna 7B        | 7B   | MHA       | NVIDIA L4 (24GB)   |
| LLaMA 2 7B       | 7B   | MHA       | NVIDIA L4 (24GB)   |
| LLaMA 3 8B       | 8B   | GQA       | NVIDIA L4 (24GB)   |
| CodeLLaMA 34B    | 34B  | GQA       | NVIDIA A100 (80GB) |

- MHA models use context lengths of 512 (prompt) / 256 (response)
- GQA models use 1024 / 512 context lengths
- All models are loaded with pre-quantized AWQ INT4 weights

---

### Serving Traces

- **[Azure LLM Inference Trace (2023)](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)**  
  Public dataset capturing anonymized request logs from Azure’s cloud LLM deployments. Includes timestamps, prompt lengths, and output sizes.  
  Used in our experiments with a 72-second segment and downsampled by a factor of **4.75×** for single-GPU simulation.

- **[BurstGPT Trace](https://github.com/HPMLL/BurstGPT)**  
  Real-world workload trace collected from an academic campus, reflecting bursty traffic patterns from students and faculty using LLM-integrated tools.  
  We extract a 72-second window and apply **1.75× downsampling** to simulate high-load serving pressure.

> ⏱ Both traces are paired with sampled task inputs (see below) to build complete timestamped evaluation workloads.

---

### Datasets

| Dataset     | Task                        | Source                                                                 |
|-------------|-----------------------------|------------------------------------------------------------------------|
| **GovReport** | Long-form summarization     | [HuggingFace](https://huggingface.co/datasets/launch/gov_report)       |
| **QMSum**     | Query-based summarization   | [GitHub](https://github.com/Yale-LILY/QMSum)                           |
| **DuReader**  | Open-domain QA (EN)         | [GitHub](https://github.com/baidu/DuReader)                            |
| **Multi-News**| Multi-document summarization| [GitHub](https://github.com/Alex-Fabbri/Multi-News)                    |

Each incoming request from the trace is assigned a sampled context from these datasets to form a complete, content-rich workload for LLM evaluation.


