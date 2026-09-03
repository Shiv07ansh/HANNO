# HANNO — Hardware-Aware Neural Network Optimization for TinyML

**HANNO** (Hardware-Aware Neural Network Optimization) is a TinyML optimization and deployment pipeline for evaluating neural-network models under embedded resource constraints.

The project explores the trade-offs between **model accuracy, model size, memory requirements, and inference performance** when converting conventional neural networks into deployment-ready TinyML models.

The current implementation focuses on **Post-Training Quantization (PTQ), TensorFlow Lite conversion, model inspection, and TensorFlow Lite Micro deployment validation**. The next stage is to build a reproducible benchmarking system that quantitatively compares model variants across accuracy, latency, memory, and storage requirements.

---

## Project Goal

Deploying a neural network on a microcontroller is not simply a matter of making the model smaller.

A useful TinyML model must satisfy multiple constraints simultaneously:

```text
                    Model
                      │
                      ▼
              ┌───────────────┐
              │  Optimization │
              │  Quantization │
              └───────┬───────┘
                      │
                      ▼
               Model Variants
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Accuracy Test       TFLM Runtime
             │                 │
             │          ┌──────┴──────┐
             │          ▼             ▼
             │       Memory        Latency
             │
             └────────┬──────────────┘
                      ▼
                Trade-off Analysis
                      │
                      ▼
              Hardware Constraints
                      │
                      ▼
               Deployment Choice
```

The long-term objective of HANNO is:

> **Given a trained model, evaluation data, and a target microcontroller, determine which model configuration provides the best achievable accuracy while satisfying the hardware constraints.**

---

# Current Status

### Implemented

* [x] TensorFlow/Keras model pipeline
* [x] FP32 TensorFlow Lite conversion
* [x] Full-integer INT8 Post-Training Quantization
* [x] TFLite FlatBuffer inspection
* [x] TFLite Micro integration
* [x] C++ embedded inference runtime
* [x] INT8 tensor validation
* [x] Tensor arena measurement
* [x] C-array model export for embedded deployment
* [x] Reproducible TFLite Micro build using Bazel
* [x] Cross-runtime validation between TFLite and TFLite Micro

### In Progress

* [ ] Reproducible FP32 vs INT8 accuracy evaluation
* [ ] Automated latency benchmarking
* [ ] Automated tensor-arena benchmarking
* [ ] Model-size comparison
* [ ] Machine-readable benchmark results
* [ ] FP32/INT8 trade-off matrix

### Planned

* [ ] HANNO command-line interface
* [ ] Hardware capability database
* [ ] Hardware compatibility analysis
* [ ] Deployment constraints
* [ ] Automated model configuration search
* [ ] Pareto-optimal model selection
* [ ] ESP32-S3 device benchmarking
* [ ] Qt-based GUI
* [ ] Hardware-aware optimization

---

# Current Architecture

```text
                    Trained Model
                         │
                         ▼
                ┌─────────────────┐
                │ Python Pipeline │
                └────────┬────────┘
                         │
                  FP32 TFLite Model
                         │
                         ▼
                ┌─────────────────┐
                │ INT8 Quantizer  │
                └────────┬────────┘
                         │
                    INT8 Model
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Python TFLite          C++ TFLite Micro
              │                     │
              │                     ▼
              │               Runtime Validation
              │                     │
              ▼                     ▼
          Accuracy              Memory / Runtime
              │                     │
              └──────────┬──────────┘
                         ▼
                  Benchmarking
                         │
                         ▼
                  Trade-off Analysis
                         │
                         ▼
                Hardware Deployment
```

---

# Current Model

The current demonstration model operates on grayscale `28 × 28` inputs and produces 10 output classes.

```text
Input:
    [1, 28, 28, 1]

Output:
    [1, 10]
```

The generated INT8 TensorFlow Lite model is approximately **261 KB**, while the FP32 model is approximately **1.02 MB**.

The current TFLite Micro tensor arena requirement is approximately **16.45 KB**.

These numbers describe the current demonstration model and are not yet a complete device-level memory or performance benchmark.

---

# Pipeline Components

## 1. Model Conversion & Quantization

`src/tinyml_pipeline.py`

The Python pipeline currently:

1. Builds/trains the baseline model.
2. Converts the model to TensorFlow Lite.
3. Performs full-integer INT8 Post-Training Quantization.
4. Generates `.tflite` model artifacts.
5. Generates the C-array representation required for embedded deployment.

The resulting artifacts include:

```text
model_fp32.tflite
model_int8.tflite
model_data.h
```

---

## 2. Model Inspection

HANNO independently inspects the generated TensorFlow Lite FlatBuffer to verify model metadata before deployment.

For the current INT8 model:

```text
Input
    Shape: [1, 28, 28, 1]
    Type:  INT8
    Elements: 784
    Bytes: 784

Output
    Shape: [1, 10]
    Type:  INT8
    Elements: 10
    Bytes: 10
```

This provides an independent sanity check between the generated model and the embedded runtime.

---

## 3. TFLite Micro Runtime

`main.cpp`

The C++ runtime validates that the generated model can actually execute through **TensorFlow Lite for Microcontrollers**.

The current runtime:

* loads the embedded FlatBuffer model
* creates a TFLite Micro interpreter
* registers the required operators
* allocates the tensor arena
* validates INT8 input/output tensors
* populates an INT8 input buffer
* invokes the model
* reads and dequantizes the output
* determines the predicted class
* explicitly manages interpreter and tensor-arena lifetime

The current tensor arena is allocated as:

```cpp
constexpr size_t kTensorArenaSize = 1024 * 1024;
```

The **1 MB allocation is a capacity**, not the model's actual memory requirement. The current model requires approximately **16.45 KB of tensor arena memory**.

---

# TFLite Micro Compatibility Investigation

One of the major engineering problems encountered during development involved a mismatch between the model's actual tensor types and the types exposed by the original TFLite Micro runtime.

The TensorFlow Lite FlatBuffer correctly reported:

```text
INT8 = enum 9
```

while the original runtime reported:

```text
FLOAT32 = enum 1
```

This occurred even though:

```text
AllocateTensors() → kTfLiteOk
Invoke()         → kTfLiteOk
```

A minimal reproduction was created to isolate the problem from the application code.

The investigation verified:

```text
                 INT8 Model
                     │
                     ▼
              FlatBuffer Schema
                     │
                     │  INT8
                     ▼
             Original TFLM Build
                     │
                     │  FLOAT32
                     ▼
                 Mismatch
```

The model itself was independently verified to be correct.

The original environment used a prebuilt `libtensorflow-microlite.a` together with headers from the local TFLite Micro source tree. TFLite Micro was subsequently rebuilt from the source tree using Bazel.

The rebuilt runtime correctly reported:

```text
FlatBuffer input type: 9

Runtime input type:  9
Runtime input bytes: 784

Runtime output type: 9
Runtime output bytes: 10
```

The actual inference pipeline then successfully executed the INT8 model.

### Why this matters

This investigation established a reproducible debugging baseline:

```text
INT8 FlatBuffer
      ↓
INT8 TFLM runtime
      ↓
INT8 tensors
      ↓
Successful inference
```

It also demonstrated the importance of validating both the **model artifact** and the **runtime implementation** when debugging embedded ML systems.

The detailed investigation is documented separately in:

```text
docs/TFLM_COMPATIBILITY.md
```

---

# Benchmarking Roadmap

The next major component of HANNO is a reproducible benchmark system.

The objective is to compare model variants using the same evaluation methodology.

| Model | Accuracy | Model Size | Tensor Arena | Latency |
| ----- | -------: | ---------: | -----------: | ------: |
| FP32  |      TBD |        TBD |          TBD |     TBD |
| INT8  |      TBD |        TBD |          TBD |     TBD |

### Accuracy

Accuracy will be measured using a held-out evaluation/test dataset.

The same samples and labels will be used when evaluating different model variants so that quantization-induced accuracy changes can be measured directly.

For example:

```text
FP32 accuracy: 98.2%
INT8 accuracy: 97.9%

Accuracy change: -0.3 percentage points
```

*Example only — not a current HANNO result.*

If valid evaluation data is unavailable, accuracy will be reported as **N/A** rather than estimated using arbitrary or random inputs.

### Model Size

Measured directly from the generated `.tflite` artifact.

### Tensor Arena

Measured through TFLite Micro.

This represents the memory required by the model's TFLM tensor arena. It should not be interpreted as total MCU SRAM usage, which also includes application buffers, stack, runtime state, and other allocations.

### Latency

The benchmark harness will execute repeated inference runs after warm-up and report statistics such as:

```text
Minimum
Mean
Median
Maximum
```

This will allow FP32 and INT8 variants to be compared under the same runtime conditions.

---

# Host Benchmark vs Embedded Deployment

HANNO separates host-side benchmarking from the final embedded deployment artifact.

### Host Benchmark

```text
model.tflite
     │
     ▼
benchmark_runner
     │
     ▼
TFLite Micro
     │
     ├── Latency
     └── Tensor Arena
```

### Embedded Deployment

```text
model.tflite
     │
     ▼
model_data.h
     │
     ▼
Firmware
     │
     ▼
MCU
```

The host benchmark is intended to provide a reproducible development and comparison environment, while final performance validation will eventually be performed on physical hardware.

---

# Long-Term Architecture

The eventual HANNO system will extend beyond simple quantization into hardware-aware model selection.

```text
                         HANNO
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        Model Engine   Benchmark Engine  Hardware Engine
             │             │             │
             │             │             ▼
             │             │        Hardware Database
             │             │
             └──────┬──────┘
                    ▼
            Optimization Engine
                    │
                    ▼
          Recommendation Engine
                    │
                    ▼
             Deployment Engine
                    │
                    ▼
                  MCU
```

The optimization problem can eventually be expressed as:

$$
\max Accuracy
$$

subject to:

$$
RAM \leq RAM_{max}
$$

$$
Flash \leq Flash_{max}
$$

$$
Latency \leq Latency_{max}
$$

This transforms HANNO from a quantization pipeline into a **hardware-aware model optimization system**.

---

# Roadmap

### Phase 1 — Foundation

* Model conversion
* INT8 quantization
* Model inspection
* TFLite Micro runtime
* Runtime compatibility debugging
* Embedded model export

**Status: Complete**

### Phase 2 — Reproducible Benchmarking

* C++ benchmark harness
* Warm-up and repeated inference
* Latency statistics
* Tensor arena measurement
* Accuracy evaluation
* Model-size measurement
* Python/C++ orchestration
* JSON/CSV result generation
* FP32 vs INT8 trade-off analysis

**Status: In Progress**

### Phase 3 — HANNO CLI

```text
hanno inspect
hanno quantize
hanno benchmark
hanno validate
hanno compare
hanno run
```

### Phase 4 — Hardware Awareness

* MCU hardware database
* RAM/Flash constraints
* Runtime/operator compatibility
* Hardware-specific deployment requirements

### Phase 5 — Model Library

* TinyML model templates
* Model metadata
* Reusable architectures
* Model comparison

### Phase 6 — Qt GUI

* Model import
* Model inspection
* Benchmark interface
* Hardware selection
* Comparison dashboards
* Deployment recommendations

### Phase 7 — Visual Model Builder

* Layer palette
* Graph editor
* Live parameter/resource estimates
* Model export

### Phase 8 — Hardware-Aware Optimization

* Candidate generation
* Automated benchmarking
* Multi-objective optimization
* Pareto analysis
* Configuration recommendation

### Phase 9 — Physical Hardware Validation

Initial target:

**ESP32-S3**

The system will eventually compare host estimates against measurements from the actual MCU.

### Phase 10+ — Advanced Optimization

Potential future techniques include:

* Dynamic-range quantization
* Full INT8 quantization
* Quantization-aware training
* Pruning
* Weight clustering
* Mixed precision
* Input-resolution optimization
* Architecture search
* Operator optimization
* Knowledge distillation
* Hardware-specific optimization
* Energy-per-inference measurement

---

# Reproducibility

HANNO is designed around a simple principle:

> **Measure first. Optimize second.**

Optimization decisions should be supported by reproducible measurements rather than assumptions about model size, memory usage, or runtime performance.

The project therefore separates:

```text
Model correctness
       ↓
Runtime correctness
       ↓
Performance measurement
       ↓
Hardware constraints
       ↓
Optimization decisions
```

This makes it possible to identify whether a change actually improves deployment characteristics rather than merely producing a smaller model.

---

# Quickstart

## Requirements

Current development environment:

* Python 3
* TensorFlow
* TensorFlow Lite
* TensorFlow Lite Micro
* C++
* Bazel/Bazelisk

## Generate the Models

```bash
python3 src/tinyml_pipeline.py
```

This generates the TensorFlow Lite model artifacts used by the deployment pipeline.

## Build the TFLite Micro Runtime

From the TFLite Micro source tree:

```bash
bazel build //tensorflow/lite/micro:inference_runner
```

The resulting executable can then be used to validate the embedded inference pipeline.

---

# Project Structure

```text
HANNO/
│
├── src/
│   └── tinyml_pipeline.py
│
├── cpp/
│   └── main.cpp
│
├── models/
│   ├── model_fp32.tflite
│   ├── model_int8.tflite
│   └── model_data.h
│
├── results/
│
├── scripts/
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── TFLM_COMPATIBILITY.md
│
└── README.md
```

The exact structure will evolve as the benchmarking and CLI layers are introduced.

---

# Design Principles

### Measurement over assumption

Resource usage and performance should be measured whenever possible.

### Reproducibility

The same model, data, runtime, and benchmark methodology should produce comparable results.

### Embedded realism

Host-side results are useful for development, but physical hardware measurements ultimately determine deployment performance.

### Separation of concerns

Model optimization, runtime execution, benchmarking, hardware constraints, and deployment are separate components.

### No fabricated metrics

If a metric has not been measured, it is reported as `TBD` or `N/A`.

---

# Status

**Current milestone: Phase 1 complete → Phase 2 benchmarking**

HANNO currently demonstrates a working path from:

```text
TensorFlow/Keras
      ↓
FP32 TFLite
      ↓
INT8 PTQ
      ↓
INT8 TFLite
      ↓
TFLite Micro
      ↓
INT8 inference
```

The immediate goal is to turn this working deployment pipeline into a **reproducible FP32/INT8 benchmarking and trade-off analysis system**.

---

## License

See `LICENSE` for project licensing information.
