#include <cstdio>
#include <cstdint>
#include <cstring>
#include <numeric>

// TFLite Micro core headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Auto-generated binary array header from Python pipeline
#include "model_data.h"

// Dynamic allocation on heap for x86_64 safety
constexpr size_t kTensorArenaSize = 1024 * 1024; // 1 MB allocation

#define TRACE_LOG(fmt, ...) \
    do { \
        std::printf("[TRACE %s:%d] " fmt "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__); \
        std::fflush(stdout); \
    } while (0)

int main() {
    TRACE_LOG("========================================================");
    TRACE_LOG("[TinyML Engine] Starting execution lifecycle pipeline...");
    TRACE_LOG("========================================================");

    // 1. Allocate Arena
    TRACE_LOG("[STEP 1/8] Allocating Tensor Arena on heap (%zu bytes)...", kTensorArenaSize);
    uint8_t* tensor_arena = new uint8_t[kTensorArenaSize];
    if (!tensor_arena) {
        TRACE_LOG("CRITICAL: Failed to allocate tensor arena heap buffer!");
        return -1;
    }
    TRACE_LOG("[+] Tensor arena allocated at memory address: %p", static_cast<void*>(tensor_arena));

    // 2. Validate Model Binary
    TRACE_LOG("[STEP 2/8] Validating FlatBuffer model binary (g_model at %p)...", static_cast<const void*>(g_model));
    const tflite::Model* model = tflite::GetModel(g_model);
    if (model == nullptr) {
        TRACE_LOG("CRITICAL: tflite::GetModel(g_model) returned NULL!");
        delete[] tensor_arena;
        return -1;
    }
    
    uint32_t model_ver = model->version();
    TRACE_LOG("[+] FlatBuffer model schema version: %u (Expected: %d)", model_ver, TFLITE_SCHEMA_VERSION);
    if (model_ver != TFLITE_SCHEMA_VERSION) {
        TRACE_LOG("CRITICAL: Model schema version mismatch!");
        delete[] tensor_arena;
        return -1;
    }

    // 3. Register Op Kernels
    TRACE_LOG("[STEP 3/8] Registering built-in operator kernels...");
    static tflite::MicroMutableOpResolver<50> resolver;
    
    resolver.AddStridedSlice();
    resolver.AddShape();
    resolver.AddReshape();
    resolver.AddPack();
    resolver.AddUnpack();
    resolver.AddSqueeze();
    resolver.AddExpandDims();
    resolver.AddConcatenation();
    resolver.AddSlice();
    resolver.AddPad();
    resolver.AddPadV2();
    resolver.AddTranspose();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddTransposeConv();
    resolver.AddFullyConnected();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddLogistic();
    resolver.AddTanh();
    resolver.AddSoftmax();
    resolver.AddLeakyRelu();
    resolver.AddElu();
    resolver.AddAdd();
    resolver.AddSub();
    resolver.AddMul();
    resolver.AddDiv();
    resolver.AddMaximum();
    resolver.AddMinimum();
    resolver.AddAbs();
    resolver.AddNeg();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddCast();
    resolver.AddArgMax();
    resolver.AddArgMin();
    resolver.AddMean();
    resolver.AddSum();
    TRACE_LOG("[+] Op resolver initialized.");

    // 4. Construct Interpreter & Allocate Tensors
    TRACE_LOG("[STEP 4/8] Constructing MicroInterpreter instance...");
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    TRACE_LOG("[+] MicroInterpreter instantiated. Invoking AllocateTensors()...");

    TfLiteStatus alloc_status = interpreter.AllocateTensors();
    TRACE_LOG("[+] AllocateTensors() return status: %d (kTfLiteOk=%d)", alloc_status, kTfLiteOk);
    if (alloc_status != kTfLiteOk) {
        TRACE_LOG("CRITICAL: AllocateTensors() failed! Increase kTensorArenaSize.");
        delete[] tensor_arena;
        return -1;
    }

    size_t bytes_used = interpreter.arena_used_bytes();
    TRACE_LOG("[+] Tensor Arena Memory Allocated: %zu bytes (%.2f KB)", 
              bytes_used, static_cast<float>(bytes_used) / 1024.0f);

    // 5. Query Input and Output Handles
    TRACE_LOG("[STEP 5/8] Querying Input/Output handles...");
    TfLiteTensor* input = interpreter.input(0);
    TRACE_LOG("[+] Query input(0) handle: %p", static_cast<void*>(input));
    
    TfLiteTensor* output = interpreter.output(0);
    TRACE_LOG("[+] Query output(0) handle: %p", static_cast<void*>(output));

    if (input == nullptr || output == nullptr) {
        TRACE_LOG("CRITICAL: Input or Output handle is NULL!");
        delete[] tensor_arena;
        return -1;
    }

    TRACE_LOG("[+] Checking input->data.raw handle...");
    TRACE_LOG("[+] input->data.raw address: %p", static_cast<void*>(input->data.raw));
    
    TRACE_LOG("[+] Checking output->data.raw handle...");
    TRACE_LOG("[+] output->data.raw address: %p", static_cast<void*>(output->data.raw));

    if (input->data.raw == nullptr || output->data.raw == nullptr) {
        TRACE_LOG("CRITICAL: Raw buffer pointer for input or output tensor is NULL!");
        delete[] tensor_arena;
        return -1;
    }

    TRACE_LOG("[+] Input Tensor Byte Capacity: %zu bytes", input->bytes);
    TRACE_LOG("[+] Input Data Type Enum ID: %d", input->type);

    // Dynamic dimension verification with strict null guards
    if (input->dims != nullptr) {
        TRACE_LOG("[+] Checking input->dims size...");
        int dims_size = input->dims->size;
        TRACE_LOG("[+] input->dims->size = %d", dims_size);

        if (dims_size > 0 && input->dims->data != nullptr) {
            TRACE_LOG("[+] Reading shape dimension values...");
            for (int i = 0; i < dims_size; ++i) {
                TRACE_LOG("    Dim[%d]: %d", i, input->dims->data[i]);
            }
        } else {
            TRACE_LOG("[!] Warning: input->dims->data is NULL or size <= 0");
        }
    } else {
        TRACE_LOG("[!] Warning: input->dims pointer is NULL!");
    }

    // 6. Populate Input Data
    TRACE_LOG("[STEP 6/8] Populating input tensor data safely via memcpy...");
    size_t input_bytes = input->bytes;

    if (input->type == kTfLiteInt8) {
        TRACE_LOG("[+] Processing as INT8 input model...");
        int8_t* input_buffer = input->data.int8;
        size_t max_elements = input_bytes / sizeof(int8_t);
        TRACE_LOG("[+] Populating %zu INT8 elements...", max_elements);
        for (size_t i = 0; i < max_elements; ++i) {
            input_buffer[i] = static_cast<int8_t>((i % 256) - 128);
        }
    } else if (input->type == kTfLiteFloat32) {
        TRACE_LOG("[+] Processing as FP32 input model...");
        size_t num_floats = input_bytes / sizeof(float);
        TRACE_LOG("[+] Creating local scratch buffer for %zu float elements...", num_floats);

        float* temp_float_buf = new float[num_floats];
        TRACE_LOG("[+] Local scratch buffer allocated at %p", static_cast<void*>(temp_float_buf));

        for (size_t i = 0; i < num_floats; ++i) {
            temp_float_buf[i] = static_cast<float>(i % 100) / 100.0f;
        }

        TRACE_LOG("[+] Executing std::memcpy(%p, %p, %zu)...", 
                  static_cast<void*>(input->data.raw), 
                  static_cast<void*>(temp_float_buf), 
                  input_bytes);
        
        std::memcpy(input->data.raw, temp_float_buf, input_bytes);
        delete[] temp_float_buf;
        TRACE_LOG("[+] Scratch buffer deallocated successfully.");
    } else {
        TRACE_LOG("[!] Warning: Unhandled input data type ID %d", input->type);
    }

    // 7. Execute Inference
    TRACE_LOG("[STEP 7/8] Invoking interpreter.Invoke()...");
    TfLiteStatus invoke_status = interpreter.Invoke();
    TRACE_LOG("[+] Invoke() finished with status: %d (kTfLiteOk=%d)", invoke_status, kTfLiteOk);

    if (invoke_status != kTfLiteOk) {
        TRACE_LOG("CRITICAL: Inference invocation failed!");
        delete[] tensor_arena;
        return -1;
    }

    // 8. Output Analysis
    TRACE_LOG("[STEP 8/8] Reading and parsing output tensor results...");
    TRACE_LOG("[+] Output Tensor Byte Capacity: %zu bytes", output->bytes);
    TRACE_LOG("[+] Output Data Type Enum ID: %d", output->type);

    if (output->type == kTfLiteInt8) {
        size_t total_elements = output->bytes / sizeof(int8_t);
        int8_t* output_buffer = output->data.int8;
        int best_index = 0;
        int8_t max_score = output_buffer[0];

        size_t print_limit = (total_elements < 10) ? total_elements : 10;
        for (size_t i = 0; i < print_limit; ++i) {
            int8_t score = output_buffer[i];
            TRACE_LOG("    Output[%zu]: %d", i, score);
            if (score > max_score) {
                max_score = score;
                best_index = static_cast<int>(i);
            }
        }
        TRACE_LOG("Predicted Class Index: %d (Score: %d)", best_index, max_score);

    } else if (output->type == kTfLiteFloat32) {
        size_t total_elements = output->bytes / sizeof(float);
        float* output_buffer = output->data.f;
        int best_index = 0;
        float max_score = output_buffer[0];

        size_t print_limit = (total_elements < 10) ? total_elements : 10;
        for (size_t i = 0; i < print_limit; ++i) {
            float score = output_buffer[i];
            TRACE_LOG("    Output[%zu]: %.4f", i, score);
            if (score > max_score) {
                max_score = score;
                best_index = static_cast<int>(i);
            }
        }
        TRACE_LOG("Predicted Class Index: %d (Confidence: %.4f)", best_index, max_score);
    }

    TRACE_LOG("========================================================");
    TRACE_LOG("[TinyML Engine] Pipeline executed successfully.");
    TRACE_LOG("========================================================");

    delete[] tensor_arena;
    return 0;
}
