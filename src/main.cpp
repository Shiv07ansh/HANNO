#include <cstdio>
#include <cstdint>
#include <numeric>

// TFLite Micro core headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Auto-generated binary array header from Python pipeline
#include "model_data.h"

// Allocate working memory on heap to avoid any potential stack boundary issues
constexpr size_t kTensorArenaSize = 1024 * 1024; // 1 MB allocation

int main() {
    std::printf("========================================================\n");
    std::printf("[TinyML Engine] Initializing TFLite Micro runtime...\n");
    std::printf("========================================================\n");

    // Dynamic allocation for tensor arena guarantees heap boundary safety on x86_64
    uint8_t* tensor_arena = new uint8_t[kTensorArenaSize];

    // 1. Map the model flatbuffer from embedded C header
    const tflite::Model* model = tflite::GetModel(g_model);
    if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
        std::printf("Error: Invalid model or schema version mismatch.\n");
        delete[] tensor_arena;
        return -1;
    }

    // 2. Load ALL common built-in layer kernels
    static tflite::MicroMutableOpResolver<50> resolver;
    
    // Slicing, Reshaping & Indexing
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

    // Convolutions & Dense
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddTransposeConv();
    resolver.AddFullyConnected();

    // Pooling
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();

    // Activations
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddLogistic();
    resolver.AddTanh();
    resolver.AddSoftmax();
    resolver.AddLeakyRelu();
    resolver.AddElu();

    // Math & Quantization
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

    // 3. Construct interpreter instance
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    // 4. Allocate memory from Tensor Arena
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        std::printf("Error: AllocateTensors() failed. Increase kTensorArenaSize.\n");
        delete[] tensor_arena;
        return -1;
    }
    
    size_t bytes_used = interpreter.arena_used_bytes();
    std::printf("[+] Actual Tensor Arena Memory Used: %zu bytes (%.2f KB / %.2f MB)\n", 
                bytes_used, 
                static_cast<float>(bytes_used) / 1024.0f,
                static_cast<float>(bytes_used) / (1024.0f * 1024.0f));

    // 5. Safely query Input and Output Tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (input == nullptr || output == nullptr) {
        std::printf("Error: input(0) or output(0) handle is NULL!\n");
        delete[] tensor_arena;
        return -1;
    }

    if (input->data.raw == nullptr || output->data.raw == nullptr) {
        std::printf("Error: Tensor data raw buffer pointers are NULL!\n");
        delete[] tensor_arena;
        return -1;
    }

    std::printf("[+] Input Tensor Byte Capacity: %zu bytes\n", input->bytes);
    std::printf("[+] Input Data Type: %s\n", (input->type == kTfLiteInt8) ? "INT8" : "FP32");

    if (input->dims != nullptr) {
        std::printf("[+] Input Tensor Dimensions: [");
        for (int i = 0; i < input->dims->size; ++i) {
            std::printf("%d%s", input->dims->data[i], (i == input->dims->size - 1) ? "" : ", ");
        }
        std::printf("]\n");
    }

// 6. Populate Input Tensor safely (Alignment & Pointer Safe)
    size_t input_bytes = input->bytes;
    std::printf("[+] Populating input tensor (%zu bytes)...\n", input_bytes);

    if (input->type == kTfLiteInt8) {
        int8_t* input_buffer = input->data.int8;
        size_t max_elements = input_bytes / sizeof(int8_t);
        for (size_t i = 0; i < max_elements; ++i) {
            input_buffer[i] = static_cast<int8_t>((i % 256) - 128);
        }
    } else if (input->type == kTfLiteFloat32) {
        size_t num_floats = input_bytes / sizeof(float);
        
        // Use a heap/stack local buffer first to avoid alignment faults on input->data.f
        float* temp_float_buf = new float[num_floats];
        for (size_t i = 0; i < num_floats; ++i) {
            temp_float_buf[i] = static_cast<float>(i % 100) / 100.0f;
        }

        // Safely copy into TFLM raw tensor memory
        std::memcpy(input->data.raw, temp_float_buf, input_bytes);
        delete[] temp_float_buf;
    }

    // 7. Perform Inference Execution
    std::printf("[+] Executing model inference kernel...\n");
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        std::printf("Error: Inference execution failed with status code %d\n", invoke_status);
        delete[] tensor_arena;
        return -1;
    }

    // 8. Dynamically Parse Output Results safely
    std::printf("--------------------------------------------------------\n");
    std::printf("[+] Inference Complete. Output Details:\n");
    std::printf("[+] Output Tensor Byte Capacity: %zu bytes\n", output->bytes);

    if (output->type == kTfLiteInt8) {
        size_t total_elements = output->bytes / sizeof(int8_t);
        int8_t* output_buffer = output->data.int8;
        int best_index = 0;
        int8_t max_score = output_buffer[0];

        size_t print_limit = (total_elements < 10) ? total_elements : 10;
        for (size_t i = 0; i < print_limit; ++i) {
            int8_t score = output_buffer[i];
            std::printf("  Output[%zu]: %d\n", i, score);
            if (score > max_score) {
                max_score = score;
                best_index = static_cast<int>(i);
            }
        }
        std::printf("\nPredicted Top Output Index: %d (Score: %d)\n", best_index, max_score);

    } else if (output->type == kTfLiteFloat32) {
        size_t total_elements = output->bytes / sizeof(float);
        float* output_buffer = output->data.f;
        int best_index = 0;
        float max_score = output_buffer[0];

        size_t print_limit = (total_elements < 10) ? total_elements : 10;
        for (size_t i = 0; i < print_limit; ++i) {
            float score = output_buffer[i];
            std::printf("  Output[%zu]: %.4f\n", i, score);
            if (score > max_score) {
                max_score = score;
                best_index = static_cast<int>(i);
            }
        }
        std::printf("\nPredicted Top Output Index: %d (Confidence: %.4f)\n", best_index, max_score);
    }

    std::printf("========================================================\n");
    delete[] tensor_arena;
    return 0;
}
