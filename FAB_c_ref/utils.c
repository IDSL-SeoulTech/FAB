#define _CRT_SECURE_NO_WARNINGS
#include "utils.h"


void load_weight_double_buffer_from_txt(double* buffer, int num_elements, const char* filepath)
{
    char full_filepath[256];
    snprintf(full_filepath, sizeof(full_filepath), "./data/weight/%s.txt", filepath);

    FILE* file = fopen(full_filepath, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", full_filepath);
        exit(EXIT_FAILURE);
    }

    int read_count = 0;
    float temp = 0.0;
    for (int i = 0; i < num_elements; ++i) {
        if (fscanf(file, "%f", &temp) != 1) {
            break;
        }
        buffer[i] = (double)(temp);
        read_count++;
    }
    fclose(file);

    if (read_count != num_elements) {
        printf("Error: Mismatch between expected number of elements (%d) and actual data entries (%d) in file %s\n", num_elements, read_count, full_filepath);
        exit(EXIT_FAILURE);
    }
}
void load_activation_double_buffer_from_txt(double* buffer, int num_elements, const char* filepath)
{
    char full_filepath[256];
    snprintf(full_filepath, sizeof(full_filepath), "./data/activation/%s.txt", filepath);

    FILE* file = fopen(full_filepath, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", full_filepath);
        exit(EXIT_FAILURE);
    }

    int read_count = 0;
    float temp = 0.0;
    for (int i = 0; i < num_elements; ++i) {
        if (fscanf(file, "%f", &temp) != 1) {
            break;
        }
        buffer[i] = (double)(temp);
        read_count++;
    }
    fclose(file);

    if (read_count != num_elements) {
        printf("Error: Mismatch between expected number of elements (%d) and actual data entries (%d) in file %s\n", num_elements, read_count, full_filepath);
        exit(EXIT_FAILURE);
    }
}


void load_ref_double_buffer_from_txt(double* buffer, int num_elements, const char* filepath)
{
    char full_filepath[256];
    snprintf(full_filepath, sizeof(full_filepath), "./data/torch_data/%s.txt", filepath);

    FILE* file = fopen(full_filepath, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", full_filepath);
        exit(EXIT_FAILURE);
    }

    int read_count = 0;
    float temp = 0.0;
    for (int i = 0; i < num_elements; ++i) {
        if (fscanf(file, "%f", &temp) != 1) {
            break;
        }
        buffer[i] = (double)(temp);
        read_count++;
    }
    fclose(file);

    if (read_count != num_elements) {
        printf("Error: Mismatch between expected number of elements (%d) and actual data entries (%d) in file %s\n", num_elements, read_count, full_filepath);
        exit(EXIT_FAILURE);
    }
}


void load_ref_output_integer_buffer_from_txt(int* buffer, int num_elements, const char* filepath)
{
    char full_filepath[256];
    snprintf(full_filepath, sizeof(full_filepath), "./data/torch_data/%s.txt", filepath);

    FILE* file = fopen(full_filepath, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", full_filepath);
        exit(EXIT_FAILURE);
    }

    int read_count = 0;
    float temp = 0.0;
    for (int i = 0; i < num_elements; ++i) {
        if (fscanf(file, "%f", &temp) != 1) {
            break;
        }
        buffer[i] = (int) temp;
        read_count++;
    }
    fclose(file);

    if (read_count != num_elements) {
        printf("Error: Mismatch between expected number of elements (%d) and actual data entries (%d) in file %s\n", num_elements, read_count, full_filepath);
        exit(EXIT_FAILURE);
    }
}

void load_weight_integer_buffer_from_txt(int* buffer, int num_elements, const char* filepath)
{
    char full_filepath[256];
    snprintf(full_filepath, sizeof(full_filepath), "./data/weight/%s.txt", filepath);

    FILE* file = fopen(full_filepath, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", full_filepath);
        exit(EXIT_FAILURE);
    }
    float temp = 0;
    int read_count = 0;
    for (int i = 0; i < num_elements; ++i) {
        if (fscanf(file, "%f", &temp) != 1) {
            break;
        }
        
        buffer[i] = (int)(temp);
        read_count++;
    }
    fclose(file);

    if (read_count != num_elements) {
        printf("Error: Mismatch between expected number of elements (%d) and actual data entries (%d) in file %s\n", num_elements, read_count, full_filepath);
        exit(EXIT_FAILURE);
    }
}


void load_weight_int8_buffer_from_txt(int8_t* buffer, int num_elements, const char* filepath)
{
    char full_filepath[256];
    snprintf(full_filepath, sizeof(full_filepath), "./data/weight/%s.txt", filepath);

    FILE* file = fopen(full_filepath, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", full_filepath);
        exit(EXIT_FAILURE);
    }
    float temp = 0;
    int read_count = 0;
    for (int i = 0; i < num_elements; ++i) {
        if (fscanf(file, "%f", &temp) != 1) {
            break;
        }
        
        buffer[i] = (int8_t)(temp);
        read_count++;
    }
    fclose(file);

    if (read_count != num_elements) {
        printf("Error: Mismatch between expected number of elements (%d) and actual data entries (%d) in file %s\n", num_elements, read_count, full_filepath);
        exit(EXIT_FAILURE);
    }
}

int double_compare_buffers(double* buf1, double* buf2, int size) {
    const double epsilon = 1e-12;  // A small threshold for float comparison

    int error_num = 0;
    double error_value = 0;
    for (int i = 0; i < size; i++) {
        // printf("index %d: buf1=%.15f, buf2=%.15f, error = %.15f\n\r", i, buf1[i], buf2[i],fabs(buf1[i] - buf2[i]));
        /*
        if (isnan(buf1[i]))
        {
            printf("NAN VALUE In Buf1\n\r");
            return 0;
        }

        if (isnan(buf2[i]))
        {
            printf("NAN VALUE In Buf2\n\r");
            return 0;
        }
        */
        if (fabs(buf1[i] - buf2[i]) > epsilon) {
            printf("index : %d, buf1 : %.20f, buf2 : %.20f\n", i, buf1[i], buf2[i]);
            error_value += fabs(buf1[i] - buf2[i]);
            // printf("Mismatch at index %d: buf1=%.20f, buf2=%.20f, error = %e\n\r", i, buf1[i], buf2[i],fabs(buf1[i] - buf2[i]));
            //return 0;  // Return 0 if buffers are not identical
            error_num++;
        }
    }
    printf("AVERAGE ERROR = %e , ERROR NUM = %d \n\r", error_value / size, error_num);
    printf("--------------------------------------------------\n\r");
    return 1;  // Return 1 if buffers are identical
}


int int_compare_buffers(int* buf1, int* buf2, int size) {
    const double epsilon = 1e-12;  // A small threshold for float comparison

    int error_num = 0;
    double error_value = 0;
    for (int i = 0; i < size; i++) {
        //printf("index %d: buf1=%.15f, buf2=%.15f, error = %.15f\n\r", i, buf1[i], buf2[i],fabs(buf1[i] - buf2[i]));
        /*
        if (isnan(buf1[i]))
        {
            printf("NAN VALUE In Buf1\n\r");
            return 0;
        }

        if (isnan(buf2[i]))
        {
            printf("NAN VALUE In Buf2\n\r");
            return 0;
        }
        */
        if (fabs(buf1[i] - buf2[i]) > epsilon) {
            // printf("index : %d, buf1 : %.20f, buf2 : %.20f\n", i, buf1[i], buf2[i]);
            error_value += fabs(buf1[i] - buf2[i]);
            // printf("Mismatch at index %d: buf1=%.20f, buf2=%.20f, error = %e\n\r", i, buf1[i], buf2[i],fabs(buf1[i] - buf2[i]));
            //return 0;  // Return 0 if buffers are not identical
            error_num++;
        }
    }
    printf("AVERAGE ERROR = %e , ERROR NUM = %d \n\r", error_value / size, error_num);
    //printf("--------------------------------------------------\n\r");
    return 1;  // Return 1 if buffers are identical
}

void PWC_input_div_RTL(int8_t* real_buf, int8_t* store_buf, int i_c, int i_w) {

    int changed_idx = 0;
    for (int ih = 0; ih < i_w; ih++) {
        for (int iw = 0; iw < i_w; iw++) {
            for (int ic = 0; ic < i_c; ic++) {
                int input_idx = ih * i_w + iw + ic * i_w * i_w;
                // printf("store idx : %d , input idx : %d\n", changed_idx, input_idx);
                store_buf[changed_idx] = real_buf[input_idx];
                changed_idx++;
            }
        }
    }
}

void PWC_Weight_div_RTL(int8_t* real_buf, int8_t* store_buf, int i_c , int o_c, int PIC, int POC) {
    int changed_idx = 0;
    for (int oc = 0; oc < (int)(o_c / POC); oc++) {
        for (int ic = 0; ic < (int)(i_c / PIC); ic++) {
            for (int j = 0; j < POC; j++) {
                for (int i = 0; i < PIC; i++) {
                    int input_idx = i + j * i_c + ic * PIC + oc * POC * i_c;
                   //  printf("store index : %d, input_idx : %d\n", changed_idx, input_idx);
                    store_buf[changed_idx] = real_buf[input_idx];
                    changed_idx++;
                }
            }
        }
    }
}


void PWC_output_div_RTL(int* real_buf, int* store_buf, int i_c, int i_w, int unroll) {

    int changed_idx = 0;
    for (int ih = 0 ; ih < i_w; ih++){
        for (int ic = 0; ic < (int)(i_c / unroll); ic++) {
            for (int iw = 0; iw < i_w; iw++) {
                for (int ur = 0; ur < unroll; ur++) {
                    int input_idx = ih * i_w + iw + ur * i_w * i_w + ic * unroll * i_w * i_w;
                    // printf("store idx : %d , input idx : %d\n", changed_idx, input_idx);
                    store_buf[changed_idx] = real_buf[input_idx];
                    changed_idx++;
                }
            }
        }
    }
}

void PWC_output_div_RTL_int8(int8_t* real_buf, int8_t* store_buf, int i_c, int i_w, int unroll) {

    int changed_idx = 0;
    for (int ih = 0 ; ih < i_w; ih++){
        for (int ic = 0; ic < (int)(i_c / unroll); ic++) {
            for (int iw = 0; iw < i_w; iw++) {
                for (int ur = 0; ur < unroll; ur++) {
                    int input_idx = ih * i_w + iw + ur * i_w * i_w + ic * unroll * i_w * i_w;
                    // printf("store idx : %d , input idx : %d\n", changed_idx, input_idx);
                    store_buf[changed_idx] = real_buf[input_idx];
                    changed_idx++;
                }
            }
        }
    }
}



void reordering_int(int* ref_out, int* reordered_out, int unroll, int i_w,int i_h, int i_c) 
{
    int index = 0;
    for (int h = 0; h < i_h; h++) {
        for (int ic = 0; ic < (int)(i_c / unroll); ic++) {
            for (int w = 0; w < i_w; w++) {
                for (int ur = 0; ur < unroll; ur++) {
                    int idx = ur * i_w * i_h + w + i_w * h + ic * unroll * i_w * i_h;
                    reordered_out[index] = ref_out[idx];
                    // printf("idx : %d, origin idx : %d , stored value : %d\n", index, idx, dwc_input[idx]); 
                    index++;
                }
            }
        }
    }
}

void reordering_int8(int8_t* ref_out, int8_t* reordered_out, int unroll, int i_w, int i_h, int i_c)
{
    int index = 0;
    for (int h = 0; h < i_h; h++) {
        for (int ic = 0; ic < (int)(i_c / unroll); ic++) {
            for (int w = 0; w < i_w; w++) {
                for (int ur = 0; ur < unroll; ur++) {
                    int idx = ur * i_w * i_h + w + i_w * h + ic * unroll * i_w * i_h;
                    reordered_out[index] = ref_out[idx];
                    // printf("idx : %d, origin idx : %d , stored value : %d\n", index, idx, dwc_input[idx]); 
                    index++;
                }
            }
        }
    }
}