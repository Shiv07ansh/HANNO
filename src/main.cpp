#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

// TFLite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model
#include "model_data.h"

// ============================================================
// Configuration
// ============================================================

constexpr size_t kTensorArenaSize = 1024 * 1024;  // 1 MB

#define TRACE_LOG(fmt, ...)                                      \
    do {                                                         \
        std::printf("[TRACE %s:%d] " fmt "\n",                  \
                    __FUNCTION__, __LINE__, ##__VA_ARGS__);      \
        std::fflush(stdout);                                     \
    } while (0)


// ============================================================
// Main
// ============================================================

int main()
{
    TRACE_LOG("========================================================");
    TRACE_LOG("[TinyML Engine] Starting execution lifecycle pipeline...");
    TRACE_LOG("========================================================");


    // ========================================================
    // STEP 1
    // Allocate Tensor Arena
    // ========================================================

    TRACE_LOG("[STEP 1/8] Allocating Tensor Arena on heap (%zu bytes)...",
              kTensorArenaSize);

    uint8_t* tensor_arena = new uint8_t[kTensorArenaSize];

    if (tensor_arena == nullptr)
    {
        TRACE_LOG("CRITICAL: Failed to allocate tensor arena!");
        return -1;
    }

    TRACE_LOG("[+] Tensor arena allocated at memory address: %p",
              static_cast<void*>(tensor_arena));


    // ========================================================
    // STEP 2
    // Validate Model
    // ========================================================

    TRACE_LOG("[STEP 2/8] Validating FlatBuffer model binary "
              "(g_model at %p)...",
              static_cast<const void*>(g_model));

    const tflite::Model* model = tflite::GetModel(g_model);
    const auto* subgraph = model->subgraphs()->Get(0);

const auto* raw_input_tensor =
    subgraph->tensors()->Get(subgraph->inputs()->Get(0));

const auto* raw_output_tensor =
    subgraph->tensors()->Get(subgraph->outputs()->Get(0));

printf("\n========== RAW FLATBUFFER MODEL ==========\n");
printf("Raw input type  = %d\n",
       static_cast<int>(raw_input_tensor->type()));

printf("Raw output type = %d\n",
       static_cast<int>(raw_output_tensor->type()));

printf("Raw input dims  = ");
for (unsigned int i = 0; i < raw_input_tensor->shape()->size(); ++i) {
    printf("%d ", raw_input_tensor->shape()->Get(i));
}
printf("\n");

printf("Raw output dims = ");
for (unsigned int i = 0; i < raw_output_tensor->shape()->size(); ++i) {
    printf("%d ", raw_output_tensor->shape()->Get(i));
}
printf("\n");
printf("==========================================\n\n");
    if (model == nullptr)
    {
        TRACE_LOG("CRITICAL: tflite::GetModel(g_model) returned NULL!");
        delete[] tensor_arena;
        return -1;
    }

    uint32_t model_ver = model->version();

    TRACE_LOG("[+] FlatBuffer model schema version: %u "
              "(Expected: %d)",
              model_ver,
              TFLITE_SCHEMA_VERSION);

    if (model_ver != TFLITE_SCHEMA_VERSION)
    {
        TRACE_LOG("CRITICAL: Model schema version mismatch!");
        delete[] tensor_arena;
        return -1;
    }

    TRACE_LOG("[+] Model validation successful.");


    // ========================================================
    // STEP 3
    // Register Operators
    // ========================================================

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


    // ========================================================
    //
    // IMPORTANT:
    //
    // interpreter lives inside this scope.
    //
    // It is destroyed BEFORE tensor_arena is deleted.
    //
    // ========================================================

    int return_code = 0;

    {
        // ====================================================
        // STEP 4
        // Construct Interpreter
        // ====================================================

        TRACE_LOG("[STEP 4/8] Constructing MicroInterpreter instance...");

        tflite::MicroInterpreter interpreter(
            model,
            resolver,
            tensor_arena,
            kTensorArenaSize
        );

        TRACE_LOG("[+] MicroInterpreter instantiated.");
        TRACE_LOG("[+] Invoking AllocateTensors()...");

        TfLiteStatus alloc_status = interpreter.AllocateTensors();

        TRACE_LOG("[+] AllocateTensors() return status: %d "
                  "(kTfLiteOk=%d)",
                  alloc_status,
                  kTfLiteOk);

        if (alloc_status != kTfLiteOk)
        {
            TRACE_LOG("CRITICAL: AllocateTensors() failed!");
            TRACE_LOG("Increase kTensorArenaSize.");
            return_code = -1;
        }
        else
        {
            size_t bytes_used =
                interpreter.arena_used_bytes();

            TRACE_LOG(
                "[+] Tensor Arena Memory Allocated: %zu bytes "
                "(%.2f KB)",
                bytes_used,
                static_cast<float>(bytes_used) / 1024.0f
            );


            // ====================================================
            // STEP 5
            // Input / Output
            // ====================================================

            TRACE_LOG("[STEP 5/8] Querying Input/Output handles...");

            TfLiteTensor* input = interpreter.input(0);
            TfLiteTensor* output = interpreter.output(0);

            if (input == nullptr)
            {
                TRACE_LOG("CRITICAL: Input tensor is NULL!");
                return_code = -1;
            }
            else if (output == nullptr)
            {
                TRACE_LOG("CRITICAL: Output tensor is NULL!");
                return_code = -1;
            }
            else
            {
                TRACE_LOG("[+] Input Bytes: %zu",
                          input->bytes);

                TRACE_LOG("[+] Input Type Enum: %d",
                          input->type);

                TRACE_LOG("[+] Output Bytes: %zu",
                          output->bytes);

                TRACE_LOG("[+] Output Type Enum: %d",
                          output->type);


                // ====================================================
                // STEP 6
                // Populate Input
                // ====================================================

                TRACE_LOG("[STEP 6/8] Populating input tensor data...");

                if (input->type == kTfLiteFloat32)
                {
                    float* in_buffer =
                        interpreter.typed_input_tensor<float>(0);

                    if (in_buffer == nullptr)
                    {
                        TRACE_LOG(
                            "CRITICAL: "
                            "typed_input_tensor<float>(0) returned NULL!"
                        );

                        return_code = -1;
                    }
                    else
                    {
                        size_t num_elements =
                            input->bytes / sizeof(float);

                        for (size_t i = 0;
                             i < num_elements;
                             ++i)
                        {
                            in_buffer[i] = 0.5f;
                        }

                        TRACE_LOG(
                            "[+] Successfully populated "
                            "%zu FLOAT32 elements.",
                            num_elements
                        );
                    }
                }
                else if (input->type == kTfLiteInt8)
                {
                    int8_t* in_buffer =
                        interpreter.typed_input_tensor<int8_t>(0);

                    if (in_buffer == nullptr)
                    {
                        TRACE_LOG(
                            "CRITICAL: "
                            "typed_input_tensor<int8_t>(0) returned NULL!"
                        );

                        return_code = -1;
                    }
                    else
                    {
                        size_t num_elements =
                            input->bytes;

                        for (size_t i = 0;
                             i < num_elements;
                             ++i)
                        {
                            in_buffer[i] = 0;
                        }

                        TRACE_LOG(
                            "[+] Successfully populated "
                            "%zu INT8 elements.",
                            num_elements
                        );
                    }
                }
                else if (input->type == kTfLiteUInt8)
                {
                    uint8_t* in_buffer =
                        interpreter.typed_input_tensor<uint8_t>(0);

                    if (in_buffer == nullptr)
                    {
                        TRACE_LOG(
                            "CRITICAL: "
                            "typed_input_tensor<uint8_t>(0) returned NULL!"
                        );

                        return_code = -1;
                    }
                    else
                    {
                        size_t num_elements =
                            input->bytes;

                        for (size_t i = 0;
                             i < num_elements;
                             ++i)
                        {
                            in_buffer[i] = 0;
                        }

                        TRACE_LOG(
                            "[+] Successfully populated "
                            "%zu UINT8 elements.",
                            num_elements
                        );
                    }
                }
                else
                {
                    TRACE_LOG(
                        "CRITICAL: Unsupported input type: %d",
                        input->type
                    );

                    return_code = -1;
                }


                // ====================================================
                // STEP 7
                // Invoke
                // ====================================================

                if (return_code == 0)
                {
                    TRACE_LOG(
                        "[STEP 7/8] Invoking interpreter.Invoke()..."
                    );

                    TfLiteStatus invoke_status =
                        interpreter.Invoke();

                    TRACE_LOG(
                        "[+] Invoke() return status: %d "
                        "(kTfLiteOk=%d)",
                        invoke_status,
                        kTfLiteOk
                    );

                    if (invoke_status != kTfLiteOk)
                    {
                        TRACE_LOG(
                            "CRITICAL: Invoke() failed!"
                        );

                        return_code = -1;
                    }
                }


                // ====================================================
                // STEP 8
                // Read Output
                // ====================================================

                if (return_code == 0)
                {
                    TRACE_LOG(
                        "[STEP 8/8] Reading and parsing "
                        "output tensor results..."
                    );

                    size_t num_output_elements = 0;

                    if (output->type == kTfLiteFloat32)
                    {
                        num_output_elements =
                            output->bytes / sizeof(float);
                    }
                    else if (output->type == kTfLiteInt8 ||
                             output->type == kTfLiteUInt8)
                    {
                        num_output_elements =
                            output->bytes;
                    }
                    else
                    {
                        TRACE_LOG(
                            "CRITICAL: Unsupported output type: %d",
                            output->type
                        );

                        return_code = -1;
                    }


                    if (return_code == 0)
                    {
                        TRACE_LOG(
                            "[+] Accessible Output Elements: %zu",
                            num_output_elements
                        );

                        if (num_output_elements == 0)
                        {
                            TRACE_LOG(
                                "CRITICAL: Output contains zero elements!"
                            );

                            return_code = -1;
                        }
                        else
                        {
                            std::vector<float> real_scores(
                                num_output_elements,
                                0.0f
                            );

                            int best_class = -1;
                            float max_score = -1e30f;


                            // ========================================
                            // FLOAT32 OUTPUT
                            // ========================================

                            if (output->type == kTfLiteFloat32)
                            {
                                const float* out_floats =
                                    interpreter
                                        .typed_output_tensor<float>(0);

                                if (out_floats == nullptr)
                                {
                                    TRACE_LOG(
                                        "CRITICAL: "
                                        "typed_output_tensor<float>(0) "
                                        "returned NULL!"
                                    );

                                    return_code = -1;
                                }
                                else
                                {
                                    for (size_t i = 0;
                                         i < num_output_elements;
                                         ++i)
                                    {
                                        real_scores[i] =
                                            out_floats[i];

                                        if (real_scores[i] >
                                            max_score)
                                        {
                                            max_score =
                                                real_scores[i];

                                            best_class =
                                                static_cast<int>(i);
                                        }
                                    }
                                }
                            }


                            // ========================================
                            // INT8 OUTPUT
                            // ========================================

                            else if (output->type == kTfLiteInt8)
                            {
                                const int8_t* out_data =
                                    interpreter
                                        .typed_output_tensor<int8_t>(0);

                                if (out_data == nullptr)
                                {
                                    TRACE_LOG(
                                        "CRITICAL: "
                                        "typed_output_tensor<int8_t>(0) "
                                        "returned NULL!"
                                    );

                                    return_code = -1;
                                }
                                else
                                {
                                    float scale =
                                        output->params.scale;

                                    int zero_point =
                                        output->params.zero_point;

                                    if (scale == 0.0f)
                                    {
                                        scale = 1.0f;
                                    }

                                    for (size_t i = 0;
                                         i < num_output_elements;
                                         ++i)
                                    {
                                        real_scores[i] =
                                            (
                                                static_cast<float>(
                                                    out_data[i]
                                                )
                                                -
                                                static_cast<float>(
                                                    zero_point
                                                )
                                            )
                                            * scale;

                                        if (real_scores[i] >
                                            max_score)
                                        {
                                            max_score =
                                                real_scores[i];

                                            best_class =
                                                static_cast<int>(i);
                                        }
                                    }
                                }
                            }


                            // ========================================
                            // UINT8 OUTPUT
                            // ========================================

                            else if (output->type == kTfLiteUInt8)
                            {
                                const uint8_t* out_data =
                                    interpreter
                                        .typed_output_tensor<uint8_t>(0);

                                if (out_data == nullptr)
                                {
                                    TRACE_LOG(
                                        "CRITICAL: "
                                        "typed_output_tensor<uint8_t>(0) "
                                        "returned NULL!"
                                    );

                                    return_code = -1;
                                }
                                else
                                {
                                    float scale =
                                        output->params.scale;

                                    int zero_point =
                                        output->params.zero_point;

                                    if (scale == 0.0f)
                                    {
                                        scale = 1.0f;
                                    }

                                    for (size_t i = 0;
                                         i < num_output_elements;
                                         ++i)
                                    {
                                        real_scores[i] =
                                            (
                                                static_cast<float>(
                                                    out_data[i]
                                                )
                                                -
                                                static_cast<float>(
                                                    zero_point
                                                )
                                            )
                                            * scale;

                                        if (real_scores[i] >
                                            max_score)
                                        {
                                            max_score =
                                                real_scores[i];

                                            best_class =
                                                static_cast<int>(i);
                                        }
                                    }
                                }
                            }


                            // ========================================
                            // SOFTMAX
                            // ========================================

                            if (return_code == 0)
                            {
                                std::vector<float> probabilities(
                                    num_output_elements,
                                    0.0f
                                );

                                float sum_exp = 0.0f;

                                for (size_t i = 0;
                                     i < num_output_elements;
                                     ++i)
                                {
                                    probabilities[i] =
                                        std::exp(
                                            real_scores[i] -
                                            max_score
                                        );

                                    sum_exp +=
                                        probabilities[i];
                                }


                                if (!std::isfinite(sum_exp) ||
                                    sum_exp <= 0.0f)
                                {
                                    TRACE_LOG(
                                        "CRITICAL: Invalid "
                                        "softmax denominator: %f",
                                        sum_exp
                                    );

                                    return_code = -1;
                                }
                                else
                                {
                                    TRACE_LOG(
                                        "--------------------------------------------------------"
                                    );

                                    for (size_t i = 0;
                                         i < num_output_elements;
                                         ++i)
                                    {
                                        probabilities[i] /=
                                            sum_exp;

                                        TRACE_LOG(
                                            "Output[%zu]: "
                                            "Prob = %6.2f%% | "
                                            "Score = %f",
                                            i,
                                            probabilities[i] * 100.0f,
                                            real_scores[i]
                                        );
                                    }


                                    if (best_class >= 0 &&
                                        static_cast<size_t>(
                                            best_class
                                        ) < num_output_elements)
                                    {
                                        TRACE_LOG(
                                            "Predicted Class Index: %d "
                                            "(Confidence: %.2f%%)",
                                            best_class,
                                            probabilities[
                                                best_class
                                            ] * 100.0f
                                        );
                                    }
                                    else
                                    {
                                        TRACE_LOG(
                                            "CRITICAL: Invalid "
                                            "best class: %d",
                                            best_class
                                        );

                                        return_code = -1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ========================================================
        // CRITICAL:
        //
        // interpreter is still alive here.
        //
        // It will be destroyed automatically when this scope
        // ends.
        //
        // ========================================================
    }


    // ========================================================
    // NOW interpreter is destroyed.
    //
    // NOW it is safe to free tensor_arena.
    // ========================================================

    TRACE_LOG("[+] MicroInterpreter destroyed safely.");

    TRACE_LOG("[+] Releasing tensor arena...");

    delete[] tensor_arena;

    TRACE_LOG("[+] Tensor arena released.");

    TRACE_LOG("========================================================");

    if (return_code == 0)
    {
        TRACE_LOG(
            "[TinyML Engine] Pipeline executed successfully."
        );
    }
    else
    {
        TRACE_LOG(
            "[TinyML Engine] Pipeline FAILED."
        );
    }

    TRACE_LOG("========================================================");

    return return_code;
}
