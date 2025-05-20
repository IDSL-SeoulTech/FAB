#define _CRT_SECURE_NO_WARNINGS
#include "qact.h"

QACT* init_qact(int i_c, int i_h, int i_w, char* calibration_mode , char* bit_type , char* module_type)
{
    QACT* q_act = (QACT*)malloc(sizeof(QACT));
    q_act->i_c = i_c;
    q_act->i_h = i_h;
    q_act->i_w = i_w;
    q_act->i_num = i_c * i_h * i_w;
    
    // layer->obs = init_observer(in_channels, (int)out_channels / groups, kernel_size, kernel_size,calibration_mode,"int8","conv","weight",1);
	q_act->quant = init_quantizer(q_act->i_c,q_act->i_c,q_act->i_h, q_act->i_w,calibration_mode,"int8",module_type,"act");
    
    strncpy(q_act->calibration_mode,calibration_mode,sizeof(q_act->calibration_mode));
    strncpy(q_act->module_type,module_type,sizeof(q_act->module_type));
    strncpy(q_act->W_A,"act",sizeof(q_act->W_A));
    strncpy(q_act->bit_type,bit_type,sizeof(q_act->bit_type));
    
    q_act->scale_factor = (double*)calloc(q_act->i_c,sizeof(double));   // activation
    q_act->fused_scale = (double*)calloc(q_act->i_c,sizeof(double));    // sf_w0 * sf_a0 / sf_a2
    
    
    q_act->input = (double*)calloc(q_act->i_num,sizeof(double));
    q_act->output = (double*)calloc(q_act->i_num,sizeof(double));
    q_act->int_output = (int8_t*)calloc(q_act->i_num,sizeof(int8_t));

    return q_act;
}

void ACT_QUANT(QACT* q_act , double* input)
{   
    quantize(q_act->quant,input,q_act->int_output,q_act->scale_factor);
}


void Fused_ACT_Quant(QACT* q_act, int* input, double* scale_weight, char* calibration_mode)
{   int sf_bit = 20;
	int fxp_maker = pow(2,sf_bit);
	
    if(strcmp(calibration_mode,"tensor")==0)
    {
        q_act->fused_scale[0] = (double)((int)((scale_weight[0] / q_act->scale_factor[0]) * fxp_maker + 0.5)) / (fxp_maker);
    }
    else if(strcmp(calibration_mode,"channel_pre")==0)
    {
        for(int i = 0 ; i<q_act->i_c;i++)
        {
            q_act->fused_scale[i] = (double)((int)((scale_weight[0] / q_act->scale_factor[i]) * fxp_maker + 0.5)) / (fxp_maker);
        }
    }

    else if(strcmp(calibration_mode,"channel_channel")==0)
    {
        for(int i = 0 ; i<q_act->i_c;i++)
        {
            q_act->fused_scale[i] = (double)((int)((scale_weight[i] / q_act->scale_factor[i]) * fxp_maker + 0.5)) / (fxp_maker);
        }
    }

    else
    {
        for(int i = 0 ; i<q_act->i_c;i++)
        {
            q_act->fused_scale[i] = (double)((int)((scale_weight[i] / q_act->scale_factor[0]) * fxp_maker + 0.5)) / (fxp_maker);
        }

    }
    quantize_fused(q_act->i_c , q_act->i_h, q_act->i_w, input, q_act->int_output,q_act->fused_scale,calibration_mode,q_act->bit_type);
    
}
