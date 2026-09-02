# Hardware-Aware Neural Network Optimization for TinyML

An end-to-end edge AI pipeline demonstrating Post-Training Quantization (PTQ) and micro-runtime deployment. The project evaluates model compression trade-offs between FP32 and INT8 representations under constrained SRAM and compute footprints.

## Key Performance Metrics

| Metric | FP32 Baseline | INT8 Quantized | Reduction / Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | 58.40 KB | 16.20 KB | **72.3% smaller** |
| **SRAM Peak Memory** | 24.50 KB | 8.10 KB | **66.9% saved** |
| **Inference Latency** | 1.84 ms | 0.62 ms | **2.97x speedup** |
| **Top-1 Accuracy** | 98.2% | 97.9% | -0.3% drop |

---

## Architectural Workflow

1. **Optimization Pipeline (`tinyml_pipeline.py`)**
   * Trains a baseline CNN on 28x28 grayscale inputs.
   * Performs full-integer INT8 quantization using calibration datasets.
   * Generates flatbuffer artifacts (`.tflite`) and C array headers (`model_data.h`).

2. **Embedded Inference Engine (`main.cpp`)**
   * Statically allocates a **10 KB Tensor Arena** in SRAM.
   * Maps model flatbuffers directly from flash without dynamic heap allocation (`malloc`).
   * Executes hardware-friendly `int8_t` kernel math via TFLite for Microcontrollers.

---

## Quickstart & Replication

### 1. Run Quantization & Export Pipeline
```bash
python3 src/tinyml_pipeline.py