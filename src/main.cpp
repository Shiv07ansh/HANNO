#include <cstdio>
#include <cstdint>
#include <numeric>

// TFLite Micro core headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Auto-generated binary array header from Python pipeline
#include "model_data.h"

// Allocate working memory (Tensor Arena) for intermediate activations
constexpr int kTensorArenaSize = 64 * 1024; // 64 KB safety allocation
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

int main() {
    std::printf("========================================================\n");
    std::printf("[TinyML Engine] Initializing TFLite Micro runtime...\n");
    std::printf("========================================================\n");

    // 1. Map the model flatbuffer from the embedded C header
    const tflite::Model* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::printf("Error: Model schema version %d mismatch (expected %d)\n",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // 2. Load built-in layer kernels
    tflite::AllOpsResolver resolver;

    // 3. Construct interpreter instance
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    // 4. Allocate memory from Tensor Arena
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        std::printf("Error: AllocateTensors() failed. Increase kTensorArenaSize.\n");
        return -1;
    }

    // 5. Query Input Tensor Specs dynamically
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    std::printf("[+] Input Tensor Dimensions: [");
    int total_input_elements = 1;
    for (int i = 0; i < input->dims->size; ++i) {
        int dim_size = input->dims->data[i];
        // Handle batch dimension if represented as 0/negative
        if (dim_size <= 0) dim_size = 1; 
        
        total_input_elements *= dim_size;
        std::printf("%d%s", dim_size, (i == input->dims->size - 1) ? "" : ", ");
    }
    std::printf("]\n");

    std::printf("[+] Input Element Count: %d\n", total_input_elements);
    std::printf("[+] Input Data Type: %s\n", (input->type == kTfLiteInt8) ? "INT8" : "FP32");

    // 6. Populate Input Tensor dynamically regardless of shape
    if (input->type == kTfLiteInt8) {
        int8_t* input_buffer = input->data.int8;
        for (int i = 0; i < total_input_elements; ++i) {
            // Fill with deterministic dummy values bounded in INT8 range [-128, 127]
            input_buffer[i] = static_cast<int8_t>((i % 256) - 128);
        }
    } else if (input->type == kTfLiteFloat32) {
        float* input_buffer = input->data.f;
        for (int i = 0; i < total_input_elements; ++i) {
            input_buffer[i] = static_cast<float>(i % 100) / 100.0f;
        }
    } else {
        std::printf("Error: Unsupported input tensor data type.\n");
        return -1;
    }

    // 7. Perform Hardware/Simulated Inference Execution
    std::printf("[+] Executing model inference kernel...\n");
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        std::printf("Error: Inference execution failed.\n");
        return -1;
    }

    // 8. Dynamically Parse Output Results
    std::printf("--------------------------------------------------------\n");
    std::printf("[+] Inference Complete. Output Details:\n");
    
    int total_output_elements = 1;
    for (int i = 0; i < output->dims->size; ++i) {
        int dim_size = output->dims->data[i];
        if (dim_size > 0) total_output_elements *= dim_size;
    }

    std::printf("[+] Total Output Tensor Elements: %d\n", total_output_elements);

    if (output->type == kTfLiteInt8) {
        int best_index = 0;
        int8_t max_score = output->data.int8[0];

        // Print up to the first 10 outputs to keep console clean
        int print_limit = (total_output_elements < 10) ? total_output_elements : 10;
        for (int i = 0; i < print_limit; ++i) {
            int8_t score = output->data.int8[i];
            std::printf("  Output[%d]: %d\n", i, score);
            if (score > max_score) {
                max_score = score;
                best_index = i;
            }
        }
        std::printf("\nPredicted Top Output Index: %d (Score: %d)\n", best_index, max_score);

    } else if (output->type == kTfLiteFloat32) {
        int best_index = 0;
        float max_score = output->data.f[0];

        int print_limit = (total_output_elements < 10) ? total_output_elements : 10;
        for (int i = 0; i < print_limit; ++i) {
            float score = output->data.f[i];
            std::printf("  Output[%d]: %.4f\n", i, score);
            if (score > max_score) {
                max_score = score;
                best_index = i;
            }
        }
        std::printf("\nPredicted Top Output Index: %d (Confidence: %.4f)\n", best_index, max_score);
    }

    std::printf("========================================================\n");
    return 0;
}