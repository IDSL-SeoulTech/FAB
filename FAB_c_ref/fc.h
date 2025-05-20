#ifndef FC
#define FC
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct{
    
    int i_num;
    int o_num;
    int w_num;
    
    int8_t* q_weights;
    int* q_bias;
    
    double* output;
    double* fused_scale; 

}FullyConnectedLayer;

FullyConnectedLayer* init_fully_connected_layer(int i_num, int o_num);
void integer_linear(FullyConnectedLayer* layer, int8_t* input);
#endif 