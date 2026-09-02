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

## Trouble Shooting 
## TFLite Micro Runtime Compatibility Investigation

### Problem

During development of the C++ inference pipeline, the INT8 model appeared to be loaded successfully, but the runtime reported an unexpected tensor data type.

The model was explicitly generated as an INT8 TFLite model. Independent inspection of the FlatBuffer confirmed:

```text
Input:
  Shape: [1, 28, 28, 1]
  Type:  INT8
  Size:  784 bytes

Output:
  Shape: [1, 10]
  Type:  INT8
  Size:  10 bytes
```

In the original C++ environment, however, `TfLiteTensor::type` reported `kTfLiteFloat32`:

```text
Input Bytes: 784
Input Type Enum: 1

Output Bytes: 10
Output Type Enum: 1
```

This was inconsistent with the model itself.

The issue was initially confusing because `AllocateTensors()` returned `kTfLiteOk` and `Invoke()` also returned `kTfLiteOk`. The program therefore appeared to be functioning normally even though the runtime tensor metadata did not agree with the model schema.

### Independent Model Verification

The first step was to verify that the problem was not caused by an incorrectly generated or corrupted model.

The raw FlatBuffer schema was inspected directly using the TensorFlow Lite schema:

```cpp
const tflite::Model* model = tflite::GetModel(g_model);
```

The tensor metadata reported:

```text
Raw input type  = 9
Raw output type = 9
Raw input dims  = 1 28 28 1
Raw output dims = 1 10
```

TensorFlow Lite defines enum value `9` as `kTfLiteInt8`.

The embedded `model_data.h` was also verified against the generated model. The header contained exactly 261,272 model bytes, matching the size of the INT8 `.tflite` model.

Therefore the model binary itself was not the source of the type mismatch.

### Why This Was Dangerous

The mismatch was not merely cosmetic.

An INT8 input tensor contains:

```text
28 × 28 × 1 = 784 bytes
```

while a FLOAT32 tensor containing the same number of elements would require:

```text
784 × sizeof(float) = 3136 bytes
```

If application code trusts the incorrect runtime type and writes FLOAT32 values into a 784-byte INT8 tensor buffer, it can interpret or overwrite memory incorrectly.

Similarly, an INT8 output tensor contains only 10 bytes, whereas a FLOAT32 output tensor would contain:

```text
10 × sizeof(float) = 40 bytes
```

Reading an INT8 output buffer as FLOAT32 therefore produces meaningless values.

This explained the earlier behavior where inference technically returned `kTfLiteOk`, but the resulting probabilities were clearly corrupted.

### Minimal Reproduction

A small standalone TFLite Micro test program was created to remove the application logic from the investigation.

The program performed only the following operations:

1. Load the embedded FlatBuffer.
2. Inspect the raw model schema.
3. Construct `MicroMutableOpResolver`.
4. Construct `MicroInterpreter`.
5. Call `AllocateTensors()`.
6. Inspect `TfLiteTensor::type`.
7. Compare the runtime type against `kTfLiteInt8`.

This produced the decisive comparison:

```text
FlatBuffer input type: 9

Original runtime:
  Runtime input type:  1
  Runtime output type: 1

Expected:
  kTfLiteInt8 = 9
```

The minimal reproduction demonstrated that the discrepancy existed below the application-level inference code.

### Rebuilding TFLite Micro

The original application had been compiled using a prebuilt:

```text
libtensorflow-microlite.a
```

while including TFLite Micro headers from the local source tree.

This created a potentially dangerous situation: the headers used during compilation and the binary library providing the runtime implementation were not guaranteed to represent exactly the same source/build configuration.

The existing prebuilt TFLite Micro library was removed and the runtime was rebuilt directly from the current TFLite Micro source tree using Bazel.

A dedicated diagnostic target was created using the same `micro_framework` and `op_resolvers` dependencies used by TFLite Micro's own tests.

### Result After Clean Build

After rebuilding TFLite Micro from the current source, the minimal reproduction reported:

```text
FlatBuffer input type: 9
AllocateTensors: 0

Runtime input type:  9
Runtime input bytes: 784

Runtime output type: 9
Runtime output bytes: 10

Expected INT8 type: 9

TYPE MATCHES INT8.
```

The runtime now agreed with the model schema.

### Final End-to-End Verification

The same clean TFLite Micro build was then used to compile the actual C++ inference pipeline.

The complete pipeline reported:

```text
Raw input type  = 9
Raw output type = 9

Input Bytes: 784
Input Type Enum: 9

Output Bytes: 10
Output Type Enum: 9

Successfully populated 784 INT8 elements.

Invoke() return status: 0 (kTfLiteOk=0)

Accessible Output Elements: 10
```

The output was also numerically plausible rather than the corrupted FLOAT32 interpretation observed previously:

```text
Output[0]: Prob = 10.21% | Score = 0.121094
Output[1]: Prob =  9.94% | Score = 0.093750
...
Output[9]: Prob =  9.98% | Score = 0.097656
```

The interpreter was subsequently destroyed before the tensor arena was released, preventing the separate lifetime-related segmentation fault encountered earlier.

### Root Cause and Scope

The investigation established that:

* The INT8 model was valid.
* The embedded model data matched the INT8 model.
* The raw FlatBuffer schema correctly identified INT8 tensors.
* The original runtime exposed those tensors as FLOAT32.
* A clean TFLite Micro build from the current source correctly exposed them as INT8.
* The actual inference application worked correctly when linked against the clean build.

The investigation therefore points to a **TFLite Micro build/runtime compatibility problem in the original environment**, rather than an error in the INT8 model itself.

The exact historical cause of the original prebuilt-library mismatch should not be assumed without reproducing that specific build configuration. Possible causes include stale build artifacts, source/header/library version mismatch, or incompatible build configuration.

### Lessons Learned

This investigation highlighted several important principles for embedded ML development.

#### 1. Never assume the model is wrong

When model inference behaves unexpectedly, independently inspect the model artifact before modifying the inference code.

The FlatBuffer schema provided an authoritative reference for the expected tensor types and shapes.

#### 2. Verify runtime metadata

The model file and the runtime tensor are separate layers:

```text
TFLite FlatBuffer
        ↓
TFLite Micro interpreter
        ↓
TfLiteTensor
        ↓
Application buffer access
```

A valid model does not guarantee that an incorrectly built runtime will expose it correctly.

#### 3. Treat headers and libraries as a matched pair

When compiling C/C++ against a large framework, using headers from one source tree together with a prebuilt library from another build can create extremely difficult failures.

For TFLite Micro in particular, a reproducible build should preferably ensure that the source, headers, configuration, and compiled runtime originate from the same revision/build.

#### 4. Use minimal reproductions

The most useful step in this investigation was reducing the application to:

```text
Load model
    ↓
Allocate tensors
    ↓
Inspect tensor type
```

This isolated the problem from image preprocessing, quantization, inference logic, output parsing, and application-specific code.

#### 5. Check memory sizes as well as type enums

The byte counts provided another independent sanity check:

```text
INT8 input:   784 bytes
FLOAT32 input: 3136 bytes

INT8 output:   10 bytes
FLOAT32 output: 40 bytes
```

Comparing tensor type, shape, and byte size together is much more reliable than checking only one field.

#### 6. Reproduce before fixing

The clean-build diagnostic established a useful baseline:

```text
Known-good TFLM build
        ↓
INT8 model
        ↓
INT8 runtime tensors
        ↓
successful inference
```

This baseline can now be used to test future changes to HANNO.

### Verification Status

**Status: RESOLVED**

The final C++ pipeline successfully performs:

```text
INT8 TFLite model
       ↓
FlatBuffer validation
       ↓
TFLite Micro interpreter
       ↓
INT8 tensor allocation
       ↓
INT8 input population
       ↓
TFLM inference
       ↓
INT8 output
       ↓
dequantized scores
       ↓
class prediction
       ↓
safe interpreter destruction
       ↓
tensor arena release
```

This provides a clean baseline for subsequent hardware measurements and optimization experiments in HANNO.
