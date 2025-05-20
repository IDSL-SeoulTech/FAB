#include "bottleneck.h"

BottleneckBlock_exp1* init_bottleneck_block_exp1 (int i_c, int i_h, int i_w, int o_c, int stride){
	BottleneckBlock_exp1* bottleneck = (BottleneckBlock_exp1*)malloc(sizeof(BottleneckBlock_exp1));
	
	int dwc_stride = stride;
	int dwc_ic = i_c;
	int dwc_iw = i_h;
	int dwc_oc = dwc_ic;
	
	int prj_ic = dwc_oc;
	int prj_oc = o_c;
	int prj_iw = (int)(dwc_iw / dwc_stride);
	
	
	bottleneck->conv2_3x3_olc_qact = init_qact(dwc_ic, dwc_iw, dwc_iw, "tensor", "int8", "act");
	// weight -> per-channel , activation -> per-tensor
	bottleneck->conv2_3x3	= init_conv_layer(dwc_ic, dwc_iw, dwc_iw, dwc_oc, 3, 1, dwc_stride, dwc_oc);
	bottleneck->conv3_1x1_olc_qact = init_qact(prj_ic, prj_iw, prj_iw, "tensor", "int8","act");
	bottleneck->conv3_1x1	= init_conv_layer(prj_ic, prj_iw, prj_iw, prj_oc, 1, 0, 1, 1);
	bottleneck->qact_final  = init_qact(prj_oc, prj_iw, prj_iw, "tensor", "int8","act");
	return bottleneck;
}


BottleneckBlock* init_bottleneck_block(int i_c, int i_h, int i_w, int o_c, int stride) {
	
	BottleneckBlock* bottleneck = (BottleneckBlock*)malloc(sizeof(BottleneckBlock));
	bottleneck->dim = i_c * i_h * i_w;
	
	int exp_ic = i_c;
	int exp_oc = exp_ic * 6;
	int exp_iw = i_w;
	
	
	int dwc_stride = stride;
	int dwc_ic = exp_oc;
	int dwc_iw = exp_iw;
	int dwc_oc = dwc_ic;
	
	int prj_ic = dwc_oc;
	int prj_oc = o_c;
	int prj_iw = (int)(dwc_iw / dwc_stride);
	
	
	bottleneck->qact_shortcut = init_qact(exp_ic, exp_iw, exp_iw, "tensor", "int8","act");	
	
	bottleneck->conv1_1x1_olc_qact = init_qact(exp_ic, exp_iw, exp_iw, "tensor", "int8","act");
	bottleneck->conv1_1x1	= init_conv_layer(exp_ic, exp_iw, exp_iw, exp_oc, 1, 0, 1, 1);
	
	bottleneck->conv2_3x3_olc_qact = init_qact(dwc_ic, dwc_iw, dwc_iw, "tensor", "int8", "act");
	// weight -> per-channel , activation -> per-tensor
	bottleneck->conv2_3x3	= init_conv_layer(dwc_ic, dwc_iw, dwc_iw, dwc_oc, 3, 1, dwc_stride, dwc_oc);
	bottleneck->conv3_1x1_olc_qact = init_qact(prj_ic, prj_iw, prj_iw, "tensor", "int8","act");
	bottleneck->conv3_1x1	= init_conv_layer(prj_ic, prj_iw, prj_iw, prj_oc, 1, 0, 1, 1);
	bottleneck->qact_final  = init_qact(prj_oc, prj_iw, prj_iw, "tensor", "int8","act");
	bottleneck->shortcut_sf = (int32_t*)calloc(bottleneck->qact_final->i_c ,sizeof(int32_t));
	bottleneck->layer_sf 	= (int32_t*)calloc(bottleneck->qact_final->i_c ,sizeof(int32_t));
	bottleneck->shortcut_val = (int64_t*)calloc(bottleneck->qact_final->i_num,sizeof(int64_t));
	bottleneck->layer_val = (int64_t*)calloc(bottleneck->qact_final->i_num,sizeof(int64_t));

	bottleneck->conv_calib_max = (int64_t*)calloc(1, sizeof(int64_t));

	return bottleneck;
}

void make_mobilenetv2(MobileNetv2* model){

	model->qact_input = init_qact(3, 224, 224, "tensor", "int8", "act");
	
	model->stem = init_conv_layer(3, 224, 224, 32, 3, 1, 2, 1);
	
	model->stem_qact = init_qact(3, 224, 224, "tensor", "int8", "act");

	model->s1_bt0 = init_bottleneck_block_exp1(32, 112, 112, 16, 1);	
	
	model->s2_bt0 = init_bottleneck_block(16, 112, 112, 24, 2);		
	model->s2_bt1 = init_bottleneck_block(24, 56, 56, 24, 1);		

	model->s3_bt0 = init_bottleneck_block(24, 56, 56, 32, 2);
	model->s3_bt1 = init_bottleneck_block(32, 28, 28, 32, 1);
	model->s3_bt2 = init_bottleneck_block(32, 28, 28, 32, 1);

	model->s4_bt0 = init_bottleneck_block(32, 28, 28, 64, 2);
	model->s4_bt1 = init_bottleneck_block(64, 14, 14, 64, 1);
	model->s4_bt2 = init_bottleneck_block(64, 14, 14, 64, 1);
	model->s4_bt3 = init_bottleneck_block(64, 14, 14, 64, 1);
	
	model->s4_bt4 = init_bottleneck_block(64, 14, 14, 96, 1);
	model->s4_bt5 = init_bottleneck_block(96, 14, 14, 96, 1);
	model->s4_bt6 = init_bottleneck_block(96, 14, 14, 96, 1);

	model->s5_bt0 = init_bottleneck_block(96, 14, 14, 160, 2);
	model->s5_bt1 = init_bottleneck_block(160, 7, 7, 160, 1);
	model->s5_bt2 = init_bottleneck_block(160, 7, 7, 160, 1);
	model->s5_bt3 = init_bottleneck_block(160, 7, 7, 320, 1);
	
	model->qact_final_conv = init_qact(320,7,7,"tensor","int8","act");
	model->exp_1x1 = init_conv_layer(320, 7, 7, 1280, 1, 0, 1, 1);
	model->exp_1x1_qact = init_qact(1280,7,7,"tensor","int8","act");

	model->Avg_pool = init_avg_pool_layer(7,7,1280);

	model->fc_layer = init_fully_connected_layer(1280,1000);

	model->qact_final_fc = init_qact(1280, 1, 1, "tensor","int8", "act"); 
	

}


void load_mobilenetv2_params(MobileNetv2* model){
	load_qact_weights(model->qact_input, "qact_input",0);

	load_conv_weights(model->stem, "conv_1.block", 1);
	// we don't need to load qact_stem (No sf, just for output memory)
	
	load_bottleneck_weights_exp1(model->s1_bt0,"layer_1.mv2_block_0");
	
	load_bottleneck_weights(model->s2_bt0,"layer_2.mv2_block_0");

	load_bottleneck_weights(model->s2_bt1,"layer_2.mv2_block_1"); 

	load_bottleneck_weights(model->s3_bt0,"layer_3.mv2_block_0"); 
	load_bottleneck_weights(model->s3_bt1,"layer_3.mv2_block_1"); 
	load_bottleneck_weights(model->s3_bt2,"layer_3.mv2_block_2"); 
	
	load_bottleneck_weights(model->s4_bt0,"layer_4.mv2_block_0"); 
	load_bottleneck_weights(model->s4_bt1,"layer_4.mv2_block_1");
	load_bottleneck_weights(model->s4_bt2,"layer_4.mv2_block_2");
	load_bottleneck_weights(model->s4_bt3,"layer_4.mv2_block_3");
	
	load_bottleneck_weights(model->s4_bt4,"layer_4.mv2_block_4"); 
	load_bottleneck_weights(model->s4_bt5,"layer_4.mv2_block_5"); 
	load_bottleneck_weights(model->s4_bt6,"layer_4.mv2_block_6"); 
	
	load_bottleneck_weights(model->s5_bt0,"layer_5.mv2_block_0"); 
	load_bottleneck_weights(model->s5_bt1,"layer_5.mv2_block_1"); 
	load_bottleneck_weights(model->s5_bt2,"layer_5.mv2_block_2"); 
	load_bottleneck_weights(model->s5_bt3,"layer_5.mv2_block_3"); 
	
	load_qact_weights(model->qact_final_conv, "qact_final_conv",0);
	load_conv_weights(model->exp_1x1,"conv_1x1_exp.block", 0);
	load_qact_weights(model->exp_1x1_qact, "qact_final",0);
	load_linear_weights(model->fc_layer, "classifier.classifier_fc.linear");
	load_qact_weights(model->qact_final_fc, "qact_fc",0);
	
}


void Bottleneck_block_exp1_rtl(BottleneckBlock_exp1* bottleneck, QACT* qact_next){
	QConv2d(bottleneck->conv2_3x3_olc_qact->int_output, bottleneck->conv2_3x3);
	Fused_ReLU6(bottleneck->conv2_3x3, bottleneck->conv3_1x1_olc_qact,"channel");
	QConv2d(bottleneck->conv3_1x1_olc_qact->int_output, bottleneck->conv3_1x1);
	Fused_ACT_Quant(qact_next, bottleneck->conv3_1x1->q_output, bottleneck->conv3_1x1->fused_scale, "tensor");
}

void Bottleneck_block_rtl(BottleneckBlock* bottleneck, QACT* qact_next){
	
	QConv2d(bottleneck->conv1_1x1_olc_qact->int_output, bottleneck->conv1_1x1);
	Fused_ReLU6(bottleneck->conv1_1x1, bottleneck->conv2_3x3_olc_qact,"tensor");
	QConv2d(bottleneck->conv2_3x3_olc_qact->int_output, bottleneck->conv2_3x3);
	Fused_ReLU6(bottleneck->conv2_3x3, bottleneck->conv3_1x1_olc_qact, "channel");
	QConv2d(bottleneck->conv3_1x1_olc_qact->int_output, bottleneck->conv3_1x1);
	
	if(bottleneck->conv2_3x3->stride == 1){
		// Q residual 
		int8_t max_int8 = INT8_MAX;
		int8_t min_int8 = INT8_MIN;
		int sf_bit = 20;
		int fxp_maker = pow(2,sf_bit);
		bottleneck->shortcut_sf[0] = (int64_t)((bottleneck->qact_shortcut->scale_factor[0] / qact_next->scale_factor[0]) * fxp_maker + 0.5);
		bottleneck->layer_sf[0]    = (int64_t)((bottleneck->conv3_1x1->fused_scale[0] / qact_next->scale_factor[0]) * fxp_maker + 0.5);

		for(int i = 0; i < qact_next->i_num; i++)
		{ 
			bottleneck->shortcut_val[i] = bottleneck->conv1_1x1_olc_qact->int_output[i] * bottleneck->shortcut_sf[0];	// int 64
			bottleneck->layer_val[i] = bottleneck->conv3_1x1->q_output[i] * bottleneck->layer_sf[0];
			int64_t val = bottleneck->shortcut_val[i] + bottleneck->layer_val[i];
			if(val < 0){val -= 524288;}
			else{val += 524288;}
			val = val / (1<< 20); // INT64 val -> fp a
            if(val < min_int8){val = min_int8;}
            else if(val > max_int8) {val = max_int8;}
			qact_next->int_output[i] = (int8_t)(val); 
		}
	}
	else
	{
		Fused_ACT_Quant(qact_next, bottleneck->conv3_1x1->q_output, bottleneck->conv3_1x1->fused_scale,"tensor");
	}
}

void Fused_ReLU6(ConvLayer* layer, QACT* layer_qact, const char* calibration_mode){
	int sf_bit = 32;	
	int fxp_maker = pow(2,sf_bit);
	if(strcmp(calibration_mode, "tensor")==0)
	{
		layer_qact->fused_scale[0] = (double)((int64_t)((layer->fused_scale[0] / layer_qact->scale_factor[0]) * fxp_maker + 0.5)) / (fxp_maker);
	}
	
	else if(strcmp(calibration_mode, "channel")==0)
	{
		for(int i = 0 ; i < layer_qact->i_c; i++){
			layer_qact->fused_scale[i] = (double)((int64_t)((layer->fused_scale[i] / layer_qact->scale_factor[0]) * fxp_maker + 0.5)) / (fxp_maker);
		}
	}
	else
	{
		for(int i = 0 ; i < layer_qact->i_c; i++){ 
			layer_qact->fused_scale[i] = (double)((int64_t)((layer->fused_scale[0] / layer_qact->scale_factor[i]) * fxp_maker + 0.5)) / (fxp_maker);
		}
	}

	quantize_fused_ReLU6(layer_qact->i_c, layer_qact->i_h, layer_qact->i_w, layer->q_output, layer_qact->int_output, layer_qact->fused_scale,calibration_mode,layer_qact->bit_type);
}

void load_bottleneck_weights(BottleneckBlock* bottleneck, const char* prefix){
	char filepath[256];
	if(bottleneck->conv2_3x3->stride == 1){
		snprintf(filepath, sizeof(filepath), "%s.qact_shortcut",prefix);
		load_qact_weights(bottleneck->qact_shortcut, filepath, 0);
	}
	
	snprintf(filepath, sizeof(filepath), "%s.conv1_1x1_olc_qact",prefix);
	load_qact_weights(bottleneck->conv1_1x1_olc_qact, filepath, 0);
	
	snprintf(filepath, sizeof(filepath), "%s.block.exp_1x1.block", prefix);
	load_conv_weights(bottleneck->conv1_1x1, filepath, 0);

	snprintf(filepath, sizeof(filepath), "%s.qact1_1x1",prefix);
	load_qact_weights(bottleneck->conv2_3x3_olc_qact, filepath, 0);
	
	snprintf(filepath, sizeof(filepath), "%s.block.conv_3x3.block", prefix);
	load_conv_weights(bottleneck->conv2_3x3, filepath, 1);
	
	snprintf(filepath, sizeof(filepath), "%s.conv3_1x1_olc_qact",prefix);
	load_qact_weights(bottleneck->conv3_1x1_olc_qact, filepath, 0);
	
	snprintf(filepath, sizeof(filepath), "%s.block.red_1x1.block", prefix);
	load_conv_weights(bottleneck->conv3_1x1, filepath, 0);

}

void load_linear_weights(FullyConnectedLayer* layer, const char* prefix){
	
	char filepath[256];
	snprintf(filepath, sizeof(filepath),"%s.int_weight", prefix);
	load_weight_int8_buffer_from_txt(layer->q_weights, layer->w_num, filepath); 

	snprintf(filepath, sizeof(filepath),"%s.int_bias", prefix);
	load_weight_integer_buffer_from_txt(layer->q_bias, layer->o_num, filepath);

	snprintf(filepath, sizeof(filepath),"%s.bias_scale",prefix); 
	load_weight_double_buffer_from_txt(layer->fused_scale, 1 , filepath);
}



void load_bottleneck_weights_exp1(BottleneckBlock_exp1* bottleneck, const char* prefix){
	char filepath[256];
	
	snprintf(filepath, sizeof(filepath), "%s.qact1_1x1",prefix);
	load_qact_weights(bottleneck->conv2_3x3_olc_qact, filepath, 0);
	
	snprintf(filepath, sizeof(filepath), "%s.block.conv_3x3.block", prefix);
	load_conv_weights(bottleneck->conv2_3x3, filepath, 1);
	
	snprintf(filepath, sizeof(filepath), "%s.conv3_1x1_olc_qact",prefix);
	load_qact_weights(bottleneck->conv3_1x1_olc_qact, filepath, 0);
	
	snprintf(filepath, sizeof(filepath), "%s.block.red_1x1.block", prefix);
	load_conv_weights(bottleneck->conv3_1x1, filepath, 0);
}

void load_qact_weights(QACT* q_act, const char* prefix, int is_ch_wise){
	char filepath[256];
	snprintf(filepath, sizeof(filepath),"%s.quantizer.scale", prefix);
	if(is_ch_wise){load_weight_double_buffer_from_txt(q_act->scale_factor, q_act->i_c, filepath);}
	else{load_weight_double_buffer_from_txt(q_act->scale_factor, 1, filepath);}
}

void load_conv_weights(ConvLayer* layer, const char* prefix, int is_ch_wise){
	
	char filepath[256];
	
	snprintf(filepath, sizeof(filepath),"%s.conv.int_weight", prefix);
	load_weight_int8_buffer_from_txt(layer->q_weights, layer->w_num, filepath); 
	
	snprintf(filepath, sizeof(filepath),"%s.conv.int_bias", prefix);
	load_weight_integer_buffer_from_txt(layer->q_bias, layer->o_c, filepath);

	snprintf(filepath, sizeof(filepath), "%s.conv.bias_scale",prefix);
	if(is_ch_wise){load_weight_double_buffer_from_txt(layer->fused_scale, layer->o_c, filepath);}
	else{load_weight_double_buffer_from_txt(layer->fused_scale, 1, filepath);}
	
}


void shortcut(double* layer_output, double* shortcut, int num){
	for(int i = 0; i < num; i++){
		layer_output[i] += shortcut[i];
	}
}