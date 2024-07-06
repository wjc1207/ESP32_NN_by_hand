#include "esp_system.h"//
#include "esp_sleep.h"//
#include "freertos/FreeRTOS.h" //
#include "freertos/task.h" //
#include <stdio.h> 
#include <unistd.h>
#include <stdlib.h>
#include "esp_timer.h"//

#include "model.h"

float** conv1d(float** input, float*** weight, float* bias, int C_in, int L, int K, int C_out, int stride)
{
    float** output = (float**)malloc(C_out * sizeof(float*));
    for (int i = 0; i < C_out; i++)
    {
        output[i] = (float*)malloc(L * sizeof(float));
    }
    for (int i = 0; i < C_out; i++)
    {
        for (int j = 0; j < L; j++)
        {
            output[i][j] = 0;
        }
    }
    for (int i = 0; i < C_out; i++)
    {
        for (int j = 0; j < L; j += stride)
        {
            output[i][j/stride] += bias[i];
            for (int k = 0; k < C_in; k++)
            {
                for (int l = 0; l < K; l++)
                {
                    if (j + l < L)
                    {
                        output[i][j/stride] += input[k][j + l] * weight[i][k][l];
                    }
                }
            }
        }
    }

    return output;
}

float* linear(float* input, float** weight, float* bias, int C_in, int C_out)
{
    float* output = (float*)malloc(C_out * sizeof(float));
    for (int i = 0; i < C_out; i++)
    {
        output[i] = bias[i];
        for (int j = 0; j < C_in; j++)
        {
            output[i] += input[j] * weight[i][j];
        }
    }

    return output;
}

float* ReLU(float* input, int C)
{
    float* output = (float*)malloc(C * sizeof(float));
    for (int i = 0; i < C; i++)
    {
        output[i] = input[i] > 0 ? input[i] : 0;
    }

    return output;
}

float** ReLU2D(float** input, int C, int L)
{
    float** output = (float**)malloc(C * sizeof(float*));
    for (int i = 0; i < C; i++)
    {
        output[i] = (float*)malloc(L * sizeof(float));
    }
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < L; j++)
        {
            output[i][j] = input[i][j] > 0 ? input[i][j] : 0; //elementwise ReLU
        }
    }

    return output;
}

float* convert_2d_to_1d(float** input, int C, int L)
{
    float* output = (float*)malloc(C * L * sizeof(float));
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < L; j++)
        {
            output[i * L + j] = input[i][j];
        }
    }

    return output;
}

float*** convert_to_3d_dynamic(int C, int L, int K, float input[C][L][K])
{
    float*** output = (float***)malloc(C * sizeof(float**));
    for (int i = 0; i < C; i++)
    {
        output[i] = (float**)malloc(L * sizeof(float*));
        for (int j = 0; j < L; j++)
        {
            output[i][j] = (float*)malloc(K * sizeof(float));
        }
    }
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < L; j++)
        {
            for (int k = 0; k < K; k++)
            {
                output[i][j][k] = input[i][j][k];
            }
        }
    }

    return output;
}
float** convert_to_2d_dynamic(int C, int L, float input[C][L])
{
    float** output = (float**)malloc(C * sizeof(float*));
    for (int i = 0; i < C; i++)
    {
        output[i] = (float*)malloc(L * sizeof(float));
    }
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < L; j++)
        {
            output[i][j] = input[i][j];
        }
    }

    return output;
}

float* convert_to_1d_dynamic(int C, float input[C])
{
    float* output = (float*)malloc(C * sizeof(float));
    for (int i = 0; i < C; i++)
    {
        output[i] = input[i];
    }

    return output;
}


void task_main(void *pvParameters) 
{
    // get time
    int64_t start = esp_timer_get_time();


    //create input data
    float** input = (float**)malloc(CONV1D_1_C_IN * sizeof(float*));
    for (int i = 0; i < CONV1D_1_C_IN; i++)
    {
        input[i] = (float*)malloc(CONV1D_1_L * sizeof(float));
    }
    for (int i = 0; i < CONV1D_1_C_IN; i++)
    {
        for (int j = 0; j < CONV1D_1_L; j++)
        {
            input[i][j] = 1;
        }
    }
    float*** weight1_copy = convert_to_3d_dynamic(CONV1D_1_C_OUT, CONV1D_1_C_IN, CONV1D_1_K, conv1d_1_weight);
    float* bias1_copy = convert_to_1d_dynamic(CONV1D_1_C_OUT, conv1d_1_bias);
    //conv1d
    float** output = conv1d(input, weight1_copy, bias1_copy, CONV1D_1_C_IN, CONV1D_1_L, CONV1D_1_K, CONV1D_1_C_OUT, CONV1D_1_STRIDE);
    float* output1d = convert_2d_to_1d(output, CONV1D_1_C_OUT, CONV1D_1_L_OUT);
    


    float** weight2_copy = convert_to_2d_dynamic(LINEAR_1_C_OUT, LINEAR_1_C_IN, linear_1_weight);
    float* bias2_copy = convert_to_1d_dynamic(LINEAR_1_C_OUT, linear_1_bias);
    float* output2 = linear(output1d, weight2_copy, bias2_copy, LINEAR_1_C_IN, LINEAR_1_C_OUT);

    for (int i = 0; i < LINEAR_1_C_OUT; i++)
    {
        printf("%f\n", output2[i]);
    }
    int64_t end = esp_timer_get_time();
    free(weight1_copy);
    free(bias1_copy);
    free(output);
    free(output1d);
    free(weight2_copy);
    free(bias2_copy);
    free(output2);

    printf("Time: %lld\n", end - start);
    vTaskDelete(NULL);
}

void app_main() {
    // Create a task that will run "task_main"
    xTaskCreate(task_main, "main_task", 10240, NULL, 1, NULL);
}
