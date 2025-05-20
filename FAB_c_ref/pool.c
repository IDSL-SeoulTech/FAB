#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "pool.h"


GlobAvgPoolLayer* init_avg_pool_layer(int i_h, int i_w, int i_c){

    GlobAvgPoolLayer* layer = (GlobAvgPoolLayer*)malloc(sizeof(GlobAvgPoolLayer));
    layer->i_c = i_c;
    layer->i_w = i_w;
    layer->i_h = i_h;

    layer->i_num = i_c * i_w * i_h;
    
    layer->input = (double*)calloc(layer->i_num, sizeof(double));
    layer->output = (double*)calloc(layer->i_c, sizeof(double));
    
    return layer;
}

void global_avg_pool(GlobAvgPoolLayer* layer, double* input, double* output) 
{

	for (int c = 0; c < layer->i_c; c++) {
		double sum = 0;
		for (int h = 0; h < layer->i_h; h++) {
			for (int w = 0; w < layer->i_w; w++) {
				sum += input[c * layer->i_h * layer->i_w + h * layer->i_w + w];
				
			}
		}
		output[c] = sum / (layer->i_h * layer->i_w);
	}
}

void q_avg_pool(GlobAvgPoolLayer* layer, int8_t* input, double* output, double* scale_factor, QACT* qact_header){ 

    for (int c = 0; c < layer->i_c; c++) {
		int sum = 0;
		for (int h = 0; h < layer->i_h; h++) {
			for (int w = 0; w < layer->i_w; w++) {
				sum += input[c * layer->i_h * layer->i_w + h * layer->i_w + w];
				
			}
		}
		
		// Fused Avgpool with previous activation scaling factor & avg(1/49) * 64
		// output[c] = (double)sum * (scale_factor[0] / (layer->i_h * layer->i_w));
		int64_t fused_sf = (int64_t)(scale_factor[0] / qact_header->scale_factor[0] * 1.30612277984619140625 * (1<<20) + 0.5);
		
		int output = (sum )* (1<<2);	// 16.8
		// output[c] = (double)(sum / 64) ;
		int64_t temp = output * fused_sf;	// 16,28
		if(temp < 0){temp -= 134217728;}
		else{temp += 134217728;}
		temp = temp / (1<<28);
		if(temp < -127){temp = -127;}
		else if(temp > 128) {temp = 128;}
		qact_header->int_output[c] = (int8_t)(temp);
		
	}

}