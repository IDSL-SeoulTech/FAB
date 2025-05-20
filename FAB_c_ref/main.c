#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "conv.h"
#include "utils.h"
#include "qact.h"
#include "bottleneck.h"

#define TEST_NUM (100)

int main()
{
	
    printf("-------------------------MobileNet v2 Reference-------------------------\n"); 
    // Layer Initiate
    MobileNetv2 model;
	make_mobilenetv2(&model);
    load_mobilenetv2_params(&model); 

    printf("Layer Initiation Done !! \n"); 
    int Label_equal_num = 0;
    double* test_input = (double*)calloc(model.stem->i_num, sizeof(double));
    int* ref_output = (int*)calloc(TEST_NUM, sizeof(int));

    char input_path[256];
    printf("%d images Inference start !!\n", TEST_NUM);
    int index = 0;
    
    for(index = 0; index < TEST_NUM; index++){
        
        snprintf(input_path,sizeof(input_path),"/valid_output/%d",index);
        load_ref_output_integer_buffer_from_txt(ref_output,1,input_path);
        snprintf(input_path,sizeof(input_path),"/valid_input/%d",index);
        load_ref_double_buffer_from_txt(test_input,model.stem->i_num,input_path);
        
        ACT_QUANT(model.qact_input, test_input);    // PS 
        // PL start
        // printf("img sf : %lf\n", model.qact_input->scale_factor[0]);
        
        // int8_store_buf_to_txt(model.qact_input->int_output, model.stem->i_num,"stc/stc_int_input");
        // save_int8_rtl_buffer_to_txt(model.qact_input->int_output, model.stem->i_num, "stc/stc_int_input_hex_batch0",8);    // ok 
        
        QConv2d(model.qact_input->int_output, model.stem);
        Fused_ReLU6(model.stem, model.s1_bt0->conv2_3x3_olc_qact,"channel");
        Bottleneck_block_exp1_rtl(model.s1_bt0, model.s2_bt0->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s2_bt0, model.s2_bt1->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s2_bt1, model.s3_bt0->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s3_bt0, model.s3_bt1->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s3_bt1, model.s3_bt2->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s3_bt2, model.s4_bt0->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s4_bt0, model.s4_bt1->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s4_bt1, model.s4_bt2->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s4_bt2, model.s4_bt3->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s4_bt3, model.s4_bt4->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s4_bt4, model.s4_bt5->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s4_bt5, model.s4_bt6->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s4_bt6, model.s5_bt0->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s5_bt0, model.s5_bt1->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s5_bt1, model.s5_bt2->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s5_bt2, model.s5_bt3->conv1_1x1_olc_qact);
        Bottleneck_block_rtl(model.s5_bt3, model.qact_final_conv);
        QConv2d(model.qact_final_conv->int_output, model.exp_1x1);
        Fused_ReLU6(model.exp_1x1, model.exp_1x1_qact, "tensor");
        
        // input : exp_1x1_qact->int_output / output : qact_final_fc->int_output 
        q_avg_pool(model.Avg_pool, model.exp_1x1_qact->int_output, model.Avg_pool->output, model.exp_1x1_qact->scale_factor, model.qact_final_fc);
        integer_linear(model.fc_layer, model.qact_final_fc->int_output);
        double max = -99999;
        int max_index = 0;
        
        for(int i = 0; i < model.fc_layer->o_num; i++){
            if(model.fc_layer->output[i] > max){
                max = model.fc_layer->output[i];
                max_index = i;
            }
        }
        if(max_index == (int)ref_output[0]){
            Label_equal_num += 1;
        }
    }

    printf("========================================\n\r");
    printf("========================================\n\r");
    printf("========================================\n\r");
    printf("      Total Image Num = %d      \n\r",TEST_NUM);
    printf("      Correct Image Num (Compared with Torch Ref) = %d    \n\r",Label_equal_num);
    printf("      Percentage Image Num = %.2f     \n\r",(double)Label_equal_num / TEST_NUM);
    printf("========================================\n\r");
    printf("========================================\n\r");
    printf("========================================\n\r");
}
