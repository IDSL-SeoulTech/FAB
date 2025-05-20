#pragma once
#ifndef BOTTLENECK
#define BOTTLENECK
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "conv.h"
#include "qact.h"
#include "pool.h"
#include "fc.h"

typedef struct BottleneckBlock {
	int dim;
	QACT* qact_shortcut;
	QACT* conv1_1x1_olc_qact;
	ConvLayer* conv1_1x1;
	QACT* conv2_3x3_olc_qact;
	ConvLayer* conv2_3x3;
	QACT* conv3_1x1_olc_qact;	
	ConvLayer* conv3_1x1;
	QACT* qact_final;

	// shortcut fused sf
	int32_t* shortcut_sf;

	// layer fused sf
	int32_t* layer_sf;

	int64_t* shortcut_val;
	int64_t* layer_val;

	int* conv_calib_max;

} BottleneckBlock;

typedef struct BottleneckBlock_exp1 {
	int dim;
	QACT* conv2_3x3_olc_qact;
	ConvLayer* conv2_3x3;
	QACT* conv3_1x1_olc_qact;	
	ConvLayer* conv3_1x1;
	QACT* qact_final;

	int* conv_calib_max;
} BottleneckBlock_exp1;


typedef struct MobileNetv2 {
	QACT* qact_input;
	ConvLayer* stem;
	QACT* stem_qact;
	BottleneckBlock_exp1* s1_bt0;
	
	BottleneckBlock* s2_bt0;
	BottleneckBlock* s2_bt1;
	
	BottleneckBlock* s3_bt0;
	BottleneckBlock* s3_bt1;
	BottleneckBlock* s3_bt2;

	BottleneckBlock* s4_bt0;
	BottleneckBlock* s4_bt1;
	BottleneckBlock* s4_bt2;
	BottleneckBlock* s4_bt3;

	BottleneckBlock* s4_bt4;
	BottleneckBlock* s4_bt5;
	BottleneckBlock* s4_bt6;

	BottleneckBlock* s5_bt0;
	BottleneckBlock* s5_bt1;
	BottleneckBlock* s5_bt2;
	
	BottleneckBlock* s5_bt3;
	
	QACT* qact_final_conv;
	ConvLayer* exp_1x1;
	QACT* exp_1x1_qact;
	
	GlobAvgPoolLayer* Avg_pool;
	FullyConnectedLayer* fc_layer;
	QACT* qact_final_fc;
	
} MobileNetv2;

BottleneckBlock* init_bottleneck_block (int i_c, int i_h, int i_w, int o_c, int stride); 
BottleneckBlock_exp1* init_bottleneck_block_exp1 (int i_c, int i_h, int i_w, int o_c, int stride); 
void make_mobilenetv2(MobileNetv2* model);
void load_mobilenetv2_params(MobileNetv2* model);
void Bottleneck_block_exp1_rtl(BottleneckBlock_exp1* bottleneck, QACT* qact_next);
void Bottleneck_block_rtl(BottleneckBlock* bottleneck, QACT* qact_next);
void shortcut(double* layer_output, double* shortcut, int num);
void Fused_ReLU6(ConvLayer* layer, QACT* layer_qact, const char* calibration_mode);
void load_bottleneck_weights(BottleneckBlock* bottleneck, const char* prefix);
void load_bottleneck_weights_exp1(BottleneckBlock_exp1* bottleneck, const char* prefix);
void load_linear_weights(FullyConnectedLayer* layer, const char* prefix);
void load_qact_weights(QACT* q_act, const char* prefix, int is_ch_wise);
void load_conv_weights(ConvLayer* layer, const char* prefix, int is_ch_wise);
#endif