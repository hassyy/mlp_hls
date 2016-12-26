#ifndef __MLP_CONFIG_H__ 
#define __MLP_CONFIG_H__ 

#include <stdint.h>
#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_uint<8> loop8_t;
typedef ap_uint<4> loop_t;        // 4
typedef ap_uint<10> loop_sample_t; // 150

//typedef float data_t;
//typedef uint32_t data_t;
//typedef ap_fixed<32,16> data_t; // OK
//typedef ap_fixed<25,16> data_t; // OK
//typedef ap_fixed<24,16> data_t; // NG
//typedef ap_fixed<20,8> data_t; // OK
//typedef ap_fixed<16,6> data_t; // OK
//typedef ap_fixed<8,6> data_t; // NG
//typedef ap_fixed<12,6> data_t; // NG
typedef ap_fixed<16,5> data_t; // OK
//typedef ap_fixed<15,5> data_t; // NG
//typedef ap_fixed<16,4> data_t; // NG
//typedef ap_fixed<16,2> data_t; // NG
//typedef ap_fixed<15,4> data_t; // NG

static const int NUM_I_LAYER = 2; // Input Vector dimension. 2 for (X, Y)
static const int NUM_H_LAYER = 3; // Num of Hidden Layer . Default=3
static const int NUM_O_LAYER = 3; // Num of Hidden Layer
static const int NUM_REPEAT  = 10; // Num of Repeat. Default=500
static const int NUM_SAMPLE  = 150; // Num of Input Samples (x). 150


#endif


