#ifndef qact
#define qact


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "quantizer.h"
#include "conv.h"

typedef struct QACT
{
    int i_c;
    int i_w;
    int i_h;
 
    int qmax;
    int qmin;
    int i_num;

    double* scale_factor;

    double* fused_scale;

    double* input;
    double* output;
   
    int8_t* int_output;

    char calibration_mode[10];
    char bit_type[10];
    char module_type[10];
    char W_A[10]; 
    Quantizer* quant;

} QACT;

QACT* init_qact(int i_c, int i_h, int i_w, char* calibration_mode , char* bit_type , char* module_type); 
void ACT_QUANT(QACT* q_act , double* input);

void Fused_ACT_Quant(QACT* q_act,int* input, double* scale_weight, char* calibration_mode);

#endif