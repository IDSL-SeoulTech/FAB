#pragma once
#ifndef UTILS
#define UTILS
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include "bottleneck.h"

void load_weight_float_buffer_from_txt(float* buffer, int num_elements, const char* filepath);
void load_activation_float_buffer_from_txt(float* buffer, int num_elements, const char* filepath);
void load_weight_integer_buffer_from_txt(int* buffer, int num_elements, const char* filepath);
void load_weight_int8_buffer_from_txt(int8_t* buffer, int num_elements, const char* filepath);
void load_ref_double_buffer_from_txt(double* buffer, int num_elements, const char* filepath);
void load_ref_output_integer_buffer_from_txt(int* buffer, int num_elements, const char* filepath);
int float_compare_buffers(float* buf1, float* buf2, int size);
int int_compare_buffers(int* buf1, int* buf2, int size);
void PWC_input_div_RTL(int8_t* real_buf, int8_t* store_buf, int i_c, int i_w);
void PWC_Weight_div_RTL(int8_t* real_buf, int8_t* store_buf, int i_c , int o_c, int PIC, int POC);
void PWC_output_div_RTL(int* real_buf, int* store_buf, int i_c, int i_w, int unroll);
void PWC_output_div_RTL_int8(int8_t* real_buf, int8_t* store_buf, int i_c, int i_w, int unroll);
void reordering_int(int* ref_out, int* reordered_out, int unroll, int i_w,int i_h, int i_c);
void reordering_int8(int8_t* ref_out, int8_t* reordered_out, int unroll, int i_w, int i_h, int i_c);
#endif