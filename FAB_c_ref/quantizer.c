#define _CRT_SECURE_NO_WARNINGS
#include "quantizer.h"
#include <time.h>



Quantizer* init_quantizer(int i_c, int o_c, int i_h, int i_w, char* calibration_mode , char* bit_type , char* module_type, char* W_A)
{
    Quantizer* quant = (Quantizer*)malloc(sizeof(Quantizer));

    quant->i_c = i_c;
    
    quant->o_c = o_c;
    
    quant->i_h = i_h;
    quant->i_w = i_w;
    quant->i_num = i_c * i_h * i_w;

    int8_t max_int8 = INT8_MAX;
    int8_t min_int8 = INT8_MIN;

    quant->qmax = max_int8;
    quant->qmin = min_int8;

    strncpy(quant->W_A,W_A,sizeof(quant->W_A));
    strncpy(quant->calibration_mode, calibration_mode, sizeof(quant->calibration_mode));
    strncpy(quant->bit_type, bit_type, sizeof(quant->bit_type));
    strncpy(quant->module_type, module_type, sizeof(quant->module_type));
    return quant;
}



int customRound(double x) {
    int integerPart = (int)x; 
    double decimalPart = fabs(x - integerPart); 

    double diff = fabs(decimalPart-0.5);

    if (diff < 1e-10)
    {
        if (integerPart % 2 == 0) { return integerPart;} 
        else {return (x > 0) ? (integerPart + 1) : (integerPart - 1); }
    } 
    else 
    {
        int rounded = (int)(x + ((x > 0) ? 0.5 : -0.5)); 
        return rounded;
    }
}

int round_quant(double x)
{
    int integerPart = (int)x; 
    double decimalPart = fabs(x - integerPart); 

    double diff = fabs(decimalPart-0.5);

    if (diff < 1e-15)
    {
        if (integerPart % 2 == 0) { return integerPart;} 
        else {return (x > 0) ? (integerPart + 1) : (integerPart - 1); }
    } 
    else 
    {
        int rounded = (int)(x + ((x > 0) ? 0.5 : -0.5)); 
        return rounded;
    }
}


void quantize(Quantizer* quant, double* input , int8_t* output , double* scale_factor)
{
    int8_t max_int8 = INT8_MAX;
    int8_t min_int8 = INT8_MIN;
    if(strcmp(quant->calibration_mode,"tensor") == 0 )  
    {   
        
        int o_ch = 0 ;
        int q_min = 0;
        int q_max = 0;
        int i_num = quant->i_c * quant->i_h * quant->i_w;

        
        for (int i = 0 ; i < i_num ; i++)
        {

            double  value = input[i] / scale_factor[0];
            int out_value = round_quant(value);
            if(out_value < min_int8){out_value = min_int8;}
            else if(out_value > max_int8) {out_value = max_int8;}
            output[i] = (int8_t)out_value;
        }
          
    }

    else if(strcmp(quant->calibration_mode,"channel")==0) 
    {
        int out_ch = 0 ;
        
       
        for(int c = 0 ; c < (quant->i_c); c ++)
        {
            for(int h = 0 ; h < quant->i_h ; h ++ )
            {
                for(int w =0 ; w < quant->i_w ; w ++ )
                {
                    
                    int index = w + h  * quant->i_w + c * quant->i_h * quant->i_w;
                    double value = input[index] / scale_factor[c];
                    int out_value = round_quant(value);
                    if(out_value < min_int8){out_value = min_int8;}
                    else if(out_value > max_int8) {out_value = max_int8;}
                    output[index] = (int8_t) out_value;
                    
                }
            }

        }
        
    }

}

void quantize_fused_ReLU6(int i_c, int i_h, int i_w, int* input , int8_t* output , double* scale_factor,char* calibration_mode, char* bit_type)
{
    int8_t max_int8 = 0;
    int8_t min_int8 = 0;

    if(strcmp(bit_type,"int4")==0)
    {
        max_int8 = 7;
        min_int8 = -8;
    }

    else
    {
        max_int8 = INT8_MAX;
        min_int8 = 0;
    }
    

    if(strcmp(calibration_mode,"tensor")==0)
    {
        for(int i = 0; i < i_c * i_h * i_w ; i ++)
        {
            double  value = input[i] * scale_factor[0];
            int out_value = round_quant(value);
            if(out_value < min_int8){out_value = min_int8;}
            else if(out_value > max_int8) {out_value = max_int8;}
            output[i] = (int8_t) out_value;
        }
  
    }
    /////////////////////channel///////////////////////
    else 
    {
        for(int i = 0; i < i_c; i ++)
        {
            for(int j = 0 ; j < i_h*i_w ; j++)
            {

                int index = i * i_h * i_w + j ;
                double value = input[index] * scale_factor[i];
                int out_value = round_quant(value);
                if(out_value < min_int8){out_value = min_int8;}
                else if(out_value > max_int8) {out_value = max_int8;}
                output[index] = (int8_t) out_value;
            }
        }
    }

}

void dequantize(Quantizer* quant, int* input , double* output, double* scale_factor, const char* calibration_mode)
{
    

    if(strcmp(calibration_mode,"tensor")==0)
    {
        int o_len = quant->i_num;
        for ( int i = 0 ; i < o_len ; i ++)
        {output[i] = (double)(input[i] * scale_factor[0]);}
      
    }

    else if(strcmp(calibration_mode,"channel")==0)
    {
       
        for(int c = 0 ; c < quant->i_c ; c++)
        {
            for (int h = 0 ; h < quant->i_h; h++)
            {
                for(int w = 0 ; w < quant->i_w ; w++)
                {
                    int index = w + (h * quant->i_w) + (c * quant->i_w * quant->i_h);
                    double value = (double)(input[index] * scale_factor[c]);
                    output[index] = value;
                }
            }
        }
    }
    
}


void my_dequantize(Quantizer* quant, int8_t* input , double* output, double* scale_factor, const char* calibration_mode)
{
    if(strcmp(calibration_mode,"tensor")==0)
    {
        int o_len = quant->i_num;
        for ( int i = 0 ; i < o_len ; i ++)
        {   output[i] = (double)(input[i] * scale_factor[0]);}
      
    }

    else if(strcmp(calibration_mode,"channel")==0)
    {
       
        for(int c = 0 ; c < quant->i_c ; c++)
        {
            for (int h = 0 ; h < quant->i_h; h++)
            {
                for(int w = 0 ; w < quant->i_w ; w++)
                {
                    int index = w + (h * quant->i_w) + (c * quant->i_w * quant->i_h);
                    double value = (double)(input[index] * scale_factor[c]);
                    output[index] = value;
                }
            }
        }
    }
    
}

void quantize_fused(int i_c, int i_h, int i_w, int* input , int8_t* output , double* scale_factor,char* calibration_mode, char* bit_type)
{
    int8_t max_int8 = 0;
    int8_t min_int8 = 0;

    if(strcmp(bit_type,"int4")==0)
    {
        max_int8 = 7;
        min_int8 = -8;
    }

    else
    {
        max_int8 = INT8_MAX;
        min_int8 = INT8_MIN;
    }
    

    if(strcmp(calibration_mode,"tensor")==0)
    {
        for(int i = 0; i < i_c * i_h * i_w ; i ++)
        {
            double  value = input[i] * scale_factor[0];
            int out_value = round_quant(value);
            if(out_value < min_int8){out_value = min_int8;}
            else if(out_value > max_int8) {out_value = max_int8;}
            output[i] = (int8_t) out_value;
        }
  
    }
    /////////////////////channel///////////////////////
    else 
    {
        for(int i = 0; i < i_c; i ++)
        {
            for(int j = 0 ; j < i_h*i_w ; j++)
            {

                int index = i * i_h * i_w + j ;
                double value = input[index] * scale_factor[i];
                int out_value = round_quant(value);
                if(out_value < min_int8){out_value = min_int8;}
                else if(out_value > max_int8) {out_value = max_int8;}
                output[index] = (int8_t) out_value;
            }
        }
    }

}
