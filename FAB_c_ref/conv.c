#define _CRT_SECURE_NO_WARNINGS

#include "conv.h"
#include <stdint.h>


ConvLayer* init_conv_layer(int in_channels, int input_height, int input_width, int out_channels, int kernel_size, int padding, int stride, int groups) {

	ConvLayer* layer = (ConvLayer*)malloc(sizeof(ConvLayer));
	// Layer information initiate 
	layer->i_h = input_height;
	layer->i_w = input_width;
	layer->i_c = in_channels;
	layer->o_c = out_channels;
	layer->kernel_size = kernel_size;
	layer->padding = padding;
	layer->stride = stride;
	layer->o_w = (input_width + 2 * padding - kernel_size) / stride + 1;
	layer->o_h = (input_height + 2 * padding - kernel_size) / stride + 1;
	layer->groups = groups;

	layer->i_num = layer->i_h * layer->i_w * layer->i_c;
	layer->o_num = layer->o_h * layer->o_w * layer->o_c;

	layer->w_num = kernel_size * kernel_size * in_channels * out_channels / groups;

	layer->input = (int8_t*)calloc(layer->i_num, sizeof(int8_t));
	layer->q_output = (int*)calloc(layer->o_num, sizeof(int));
	
	layer->fused_scale = (double*)calloc(layer->o_c, sizeof(double));
	
	layer->q_weights = (int8_t*)calloc(layer->w_num, sizeof(int8_t));
	layer->q_bias = (int*)calloc(out_channels, sizeof(int));
	return layer;
}

void QConv2d(int8_t* i_buf, ConvLayer* layer)
{
	int_convolution(layer,i_buf,layer->q_output);	
	Q_Bias(layer,layer->q_output);
}



void int_convolution(ConvLayer* layer, int8_t* input, int* output)
{

	int output_h = (layer->i_h + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
	int output_w = (layer->i_w + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;

	// allocate memory for padded input
	int padded_h = layer->i_h + 2 * layer->padding;
	int padded_w = layer->i_w + 2 * layer->padding;
	int8_t* padded_input = (int8_t*)malloc(padded_h * padded_w * layer->i_c * sizeof(int8_t));

	// pad input with zeros
	for (int i = 0; i < layer->i_c; i++) {
		for (int j = 0; j < padded_h; j++) {
			for (int k = 0; k < padded_w; k++) {
				int idx = k + (j * padded_w) + (i * padded_h * padded_w);
				if (j < layer->padding || j >= layer->i_h + layer->padding || k < layer->padding || k >= layer->i_w + layer->padding) {
					padded_input[idx] = 0;
				}
				else {
					padded_input[idx] = input[i * layer->i_h * layer->i_w + (j - layer->padding) * layer->i_w + (k - layer->padding)];
				}
			}
		}
	}



	int groups = layer->groups;

	// Group CONV
	for (int g = 0; g < groups; g++)
	{
		int o_start = g * layer->o_c / groups;
		int o_end = (g + 1) * layer->o_c / groups;
		int i_start = g * layer->i_c / groups;
		int i_end = (g + 1) * layer->i_c / groups;

		for (int o = o_start; o < o_end; o++) {
			for (int h = 0; h < output_h; h++) {
				for (int w = 0; w < output_w; w++) {
					int sum = 0;
					for (int i = i_start; i < i_end; i++) {
						for (int j = 0; j < layer->kernel_size; j++) {
							for (int k = 0; k < layer->kernel_size; k++) {

								int ih = h * layer->stride + j;
								int iw = w * layer->stride + k;
								
								int8_t input_val = padded_input[i * padded_h * padded_w + ih * padded_w + iw];
								int weight_index = 0;

								if (groups != 1) {
									weight_index = (o * layer->i_c / groups + (i - i_start)) * layer->kernel_size * layer->kernel_size + j * layer->kernel_size + k;
								}

								else
								{
									weight_index = o * layer->i_c * layer->kernel_size * layer->kernel_size + i * layer->kernel_size * layer->kernel_size + j * layer->kernel_size + k;
								}

								int weight_val = layer->q_weights[weight_index];
								
								int value = input_val * weight_val;
								sum += value;
							}
						}
					}
					output[o * output_h * output_w + h * output_w + w] = sum;

				}
			}
		}
	}

	// free memory
	free(padded_input);
}



void Q_Bias(ConvLayer* layer, int* output)
{

	int resolution = layer->o_h * layer->o_w;

	for (int o = 0; o < layer->o_c; o++) {
		for (int r = 0; r < resolution; r++) {
			int out_idx = o * resolution + r;
			output[out_idx] += layer->q_bias[o];
			
		}
	}
}

void ReLU6(ConvLayer* layer, double* input){
	
	int resolution = layer->o_h * layer->o_w * layer->o_c;
	for(int r = 0 ; r < resolution; r++){
		if(input[r] < 0.0){input[r] = 0.0;}
		else if (input[r] > 6.0){ input[r] = 6.0;}
	}
}