#ifndef quantizer
#define quantizer

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>


typedef struct
{
    int i_c;
    int i_w;
    int i_h;
    int o_c;

    int qmax;
    int qmin;

    int i_num;

    double* scale_factor;
    double* zero_point;

    double* input;
    double* output;

    char calibration_mode[10];
    char bit_type[10];
    char module_type[10];
    char W_A[10]; 

} Quantizer;



Quantizer* init_quantizer(int i_c, int o_c, int i_h, int i_w, char* calibration_mode , char* bit_type , char* module_type, char* W_A); 
void quantize(Quantizer* quant , double* input , int8_t* output , double* scale_factor);
void quantize_fused(int i_c, int i_h, int i_w, int* input , int8_t* output , double* scale_factor,char* calibration_mode,  char* bit_type);
void dequantize(Quantizer* quant, int* input , double* output, double* scale_factor, const char* calibration_mode);
void my_dequantize(Quantizer* quant, int8_t* input , double* output, double* scale_factor, const char* calibration_mode);
void quantize_fused_ReLU6(int i_c, int i_h, int i_w, int* input , int8_t* output , double* scale_factor,char* calibration_mode, char* bit_type);
int round_quant(double x);
int customRound(double num);

#endif