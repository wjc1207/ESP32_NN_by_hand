#include "esp_system.h"//
#include "esp_sleep.h"//
#include "freertos/FreeRTOS.h" //
#include "freertos/task.h" //
#include <stdio.h> 
#include <unistd.h>
#include <stdlib.h>
#include "esp_timer.h"//

#include "model.h"

#include <inttypes.h>
#include "driver/touch_pad.h"
#include "esp_log.h"
#include "esp_task_wdt.h"

float** conv1d(float** input, float*** weight, float* bias, int C_in, int L, int K, int C_out, int stride)
{
    int L_out = (L - K) / stride + 1;
    float** output = (float**)malloc(C_out * sizeof(float*));
    for (int i = 0; i < C_out; i++)
    {
        output[i] = (float*)malloc(L_out * sizeof(float));
    }
    for (int i = 0; i < C_out; i++)
    {
        for (int j = 0; j < L_out; j++)
        {
            output[i][j] = 0;
        }
    }
    for (int i = 0; i < C_out; i++)
    {
        for (int j = 0; j < L_out; j++)
        {
            for (int k = 0; k < C_in; k++)
            {
                for (int l = 0; l < K; l++)
                {
                    if (j * stride + l < L)
                    {
                        output[i][j] += input[k][j * stride + l] * weight[i][k][l];
                    }
                }
            }
            output[i][j] += bias[i];
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


float* Net(float** input) 
{
    // get time
    int64_t start = esp_timer_get_time();

    float*** weight1_copy = convert_to_3d_dynamic(CONV1D_1_C_OUT, CONV1D_1_C_IN, CONV1D_1_K, conv1d_1_weight);
    float* bias1_copy = convert_to_1d_dynamic(CONV1D_1_C_OUT, conv1d_1_bias);
    //conv1d_1
    float** output = conv1d(input, weight1_copy, bias1_copy, CONV1D_1_C_IN, CONV1D_1_L, CONV1D_1_K, CONV1D_1_C_OUT, CONV1D_1_STRIDE);
    //ReLU
    float** output1 = ReLU2D(output, CONV1D_1_C_OUT, CONV1D_2_L);
    //conv1d_2
    float*** weight2_copy = convert_to_3d_dynamic(CONV1D_2_C_OUT, CONV1D_2_C_IN, CONV1D_2_K, conv1d_2_weight);
    float* bias2_copy = convert_to_1d_dynamic(CONV1D_2_C_OUT, conv1d_2_bias);
    float** output2 = conv1d(output1, weight2_copy, bias2_copy, CONV1D_2_C_IN, CONV1D_2_L, CONV1D_2_K, CONV1D_2_C_OUT, CONV1D_2_STRIDE);
    //ReLU
    float** output3 = ReLU2D(output2, CONV1D_2_C_OUT, CONV1D_3_L);
    //conv1d_3
    float*** weight3_copy = convert_to_3d_dynamic(CONV1D_3_C_OUT, CONV1D_3_C_IN, CONV1D_3_K, conv1d_3_weight);
    float* bias3_copy = convert_to_1d_dynamic(CONV1D_3_C_OUT, conv1d_3_bias);
    float** output4 = conv1d(output3, weight3_copy, bias3_copy, CONV1D_3_C_IN, CONV1D_3_L, CONV1D_3_K, CONV1D_3_C_OUT, CONV1D_3_STRIDE);
    //ReLU
    float** output5 = ReLU2D(output4, CONV1D_3_C_OUT, CONV1D_3_L_OUT);
    //flatten
    float* output1d = convert_2d_to_1d(output5, CONV1D_3_C_OUT, CONV1D_3_L_OUT);
    //linear_1
    float** weight4_copy = convert_to_2d_dynamic(LINEAR_1_C_OUT, LINEAR_1_C_IN, linear_1_weight);
    float* bias4_copy = convert_to_1d_dynamic(LINEAR_1_C_OUT, linear_1_bias);
    float* output6 = linear(output1d, weight4_copy, bias4_copy, LINEAR_1_C_IN, LINEAR_1_C_OUT);

    int64_t end = esp_timer_get_time();
    //free variables
    free(weight1_copy);
    free(bias1_copy);
    free(output);
    free(output1);
    free(weight2_copy);
    free(bias2_copy);
    free(output2);
    free(output3);
    free(weight3_copy);
    free(bias3_copy);
    free(output4);
    free(output5);
    free(output1d);
    free(weight4_copy);
    free(bias4_copy);

    printf("Time: %lld ", end - start);

    return output6;
}

// define argmax function
int argmax(float* input, int C)
{
    int max_index = 0;
    float max_value = input[0];
    for (int i = 1; i < C; i++)
    {
        if (input[i] > max_value)
        {
            max_value = input[i];
            max_index = i;
        }
    }

    return max_index;
}

void task_main(void *pvParameters)
{
    uint32_t touch_value;
    uint32_t touch_value_list[TOUCH_BUTTON_NUM];
    bool trigger_flag = false;
    int64_t start = 0;

    /* Wait touch sensor init done */
    vTaskDelay(100 / portTICK_PERIOD_MS);
    while(1)
    {
        for (int i = 0; i < TOUCH_BUTTON_NUM; i++) {
            touch_pad_read_raw_data(button[i], &touch_value);    // read raw data.
            touch_value_list[i] = touch_value;
            if (touch_value > THRESHOLD && trigger_flag == false) { //trigger
                start = esp_timer_get_time();
                trigger_flag = true;
            }
            vTaskDelay(1 / portTICK_PERIOD_MS);
            
        }
        if (trigger_flag) { //triggered
            float** input = (float**)malloc(CONV1D_1_C_IN * sizeof(float*));
            for (int i = 0; i < CONV1D_1_C_IN; i++)
            {
                input[i] = (float*)malloc(CONV1D_1_L * sizeof(float));
            }
            for (int i = 0; i < CONV1D_1_C_IN; i++) //use trigger data
            {
                input[i][0] = touch_value_list[i]/10000.0;
                //input[i][0] = 1;
            }
            usleep(50 * 1000); // 50ms
            for (int i = 1; i < CONV1D_1_L; i++)
            {
                for (int j = 0; j < CONV1D_1_C_IN; j++)
                {
                    touch_pad_read_raw_data(button[j], &touch_value);
                    input[j][i] = touch_value/10000.0;
                    usleep(1 * 1000);
                    //printf("%f ", input[j][i]);
                    //input[j][i] = 1;
                }
                printf("...");
                usleep(50 * 1000);
            }
            int64_t end = esp_timer_get_time();
            printf("Time: %lld\n", end - start);
            float* output = Net(input);
            free(input);
            trigger_flag = false;

            for (int i = 0; i < LINEAR_1_C_OUT; i++)
            {
                printf("%f ", output[i]);
            }

            int max_index = argmax(output, LINEAR_1_C_OUT);
            printf("Result: ");
            if (max_index == 0){
                printf("A\n");
            }
            else if (max_index == 1){
                printf("B\n");
            }
            else if (max_index == 2){
                printf("C\n");
            }
            else{
                printf("D\n");
            }
            vTaskDelay(200 / portTICK_PERIOD_MS);
        }
    }
}
void app_main() {
    /* Initialize touch pad peripheral. */
    touch_pad_init();
    for (int i = 0; i < TOUCH_BUTTON_NUM; i++) {
        touch_pad_config(button[i]);
    }

    /* Denoise setting at TouchSensor 0. */
    touch_pad_denoise_t denoise = {
        /* The bits to be cancelled are determined according to the noise level. */
        .grade = TOUCH_PAD_DENOISE_BIT4,
        .cap_level = TOUCH_PAD_DENOISE_CAP_L4,
    };
    touch_pad_denoise_set_config(&denoise);
    touch_pad_denoise_enable();
    ESP_LOGI(TAG, "Denoise function init");

    /* Enable touch sensor clock. Work mode is "timer trigger". */
    touch_pad_set_fsm_mode(TOUCH_FSM_MODE_TIMER);
    touch_pad_fsm_start();

    // Create a task that will run "task_main"
    xTaskCreate(task_main, "main_task", 8192, NULL, 1, NULL); // 8192 is the stack size
}
