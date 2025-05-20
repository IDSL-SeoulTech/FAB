#pragma once
#ifndef CONV
#define CONV

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "qact.h"

typedef struct ConvLayer {
	// Layer info
	int i_h;
	int i_w;
	int i_c;
	int o_c;
	int i_num;
	int o_h;
	int o_w;
	int o_num;
	int w_num;
	int kernel_size;
	int padding;
	int stride;
	int groups;

	int8_t* input;
	int* q_output;
	int8_t* q_weights;
	int* q_bias;
	

	double* fused_scale;

} ConvLayer;

ConvLayer* init_conv_layer(int in_channels, int input_height, int input_width, int out_channels, int kernel_size, int padding, int stride, int groups);

void QConv2d(int8_t* i_buf, ConvLayer* layer);
void int_convolution(ConvLayer* layer, int8_t* input, int* output);
void Q_Bias(ConvLayer* layer, int* output);
void ReLU6(ConvLayer* layer, double* input);
#endif