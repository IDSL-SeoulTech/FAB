#ifndef POOL
#define POOL

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "qact.h"

typedef struct{ 

    int i_h;
    int i_w;
    int i_c;
    int i_num;
    double* input;
    double* output;
} GlobAvgPoolLayer;

GlobAvgPoolLayer* init_avg_pool_layer(int i_h, int i_w, int i_c);

void q_avg_pool(GlobAvgPoolLayer* layer, int8_t* input, double* output, double* scale_factor, QACT* qact_header);
void global_avg_pool(GlobAvgPoolLayer* layer, double* input, double* output);
#endif