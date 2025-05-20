#include "fc.h"


FullyConnectedLayer* init_fully_connected_layer(int i_num, int o_num){ 

    FullyConnectedLayer* layer = (FullyConnectedLayer*)malloc(sizeof(FullyConnectedLayer));
    

    layer->i_num = i_num;  
    layer->o_num = o_num;
    layer->w_num = i_num * o_num;

    layer->q_weights = (int8_t*)calloc(layer->w_num, sizeof(int8_t));
    layer->q_bias = (int*)calloc(layer->o_num, sizeof(int)); 
    layer->fused_scale = (double*)calloc(1, sizeof(double));    // per-tensor
    
    layer->output = (double*)calloc(o_num, sizeof(double));
    
    return layer;
}



void integer_linear(FullyConnectedLayer* layer, int8_t* input){

    for(int o_ch = 0; o_ch < layer->o_num ; o_ch++){
        int sum = 0;
        for(int i_ch = 0 ; i_ch < layer->i_num; i_ch++){
            int index = i_ch + layer->i_num * o_ch;
            sum += input[i_ch] * layer->q_weights[index];
        }
        layer->output[o_ch] = (sum + layer->q_bias[o_ch]) * (layer->fused_scale[0]);
    }
}