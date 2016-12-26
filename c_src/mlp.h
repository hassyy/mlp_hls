#ifndef __MLP_H__ 
#define __MLP_H__ 

#include <cmath>
#include "mlp_config.h"

template<int in, int hid, int out, int sample>
class mlp {
	private:
		data_t 	xi1[in+1]
			, xi2[hid+1]
			, xi3[out];
		data_t 	o1[in+1]
			, o2[hid+1]
			, o3[out];
		data_t 	d2[hid+1]
			, d3[out];
		data_t 	w1[(in+1)*hid]
			, w2[(hid+1)*out];

		//sigmoid function
		data_t sigmoid(data_t x);

		//derivative of sigmoid function
		data_t d_sigmoid(data_t x);

		//calculate forward propagation of input x
		void forward(const data_t *x);

	public:
		// constructor
		mlp();
		mlp(data_t _w1[], data_t _w2[]);

		// destructor
		~mlp();

		/* train: multi layer perceptron
		 * x: train data(number of elements is in*N)
		 * t: correct label(number of elements is N)
		 * N: data size
		 * eta: learning rate */
		void train(
				const data_t *x
				, const data_t *t
				, const data_t eta=0.1
				);

		// return most probable label to the input x
		data_t predict(const data_t *x);

		void set_weight(data_t *_w1, data_t *_w2);
		void get_weight(data_t *_w1, data_t *_w2);
};


// Need to have the implementation in .h for template.

// constructor
template<int in, int hid, int out, int sample>
mlp<in, hid, out, sample>::mlp(){};

template<int in, int hid, int out, int sample>
mlp<in, hid, out, sample>::mlp(data_t *_w1, data_t *_w2){
	// initialize wait
	for(int i=0; i<(in+1)*hid; i++) w1[i] = *(_w1+i);
	for(int i=0; i<(hid+1)*out; i++) w2[i] = *(_w2+i);
}

// destructor
template<int in, int hid, int out, int sample>
mlp<in, hid, out, sample>::~mlp(){
	// No dynamic allocation
}

/* train: multi layer perceptron
 * x: train data(number of elements is in*N)
 * t: correct label(number of elements is N)
 * N: data size
 * eta: learning rate */
template<int in, int hid, int out, int sample>
void
mlp<in, hid, out, sample>::train(const data_t *x, const data_t *t, const data_t eta){
		for(loop_sample_t _sample=0; _sample<sample; _sample++){
#pragma HLS PIPELINE
			// forward propagation
			forward(x+_sample*in);

			// calculate the error of output layer
			for(loop_t j=0; j<out; j++){
#pragma HLS UNROLL
				if(*(t+_sample) == j) d3[j] = o3[j]-1;
				else d3[j] = o3[j];
			}
			// update the wait of output layer
			for(loop_t i=0; i<hid+1; i++){

#pragma HLS UNROLL
				for(loop_t j=0; j<out; j++){

#pragma HLS LOOP_FLATTEN
					w2[i*out+j] -= eta*d3[j]*o2[i];
				}
			}
			// calculate the error of hidden layer
			for(loop_t j=0; j<hid+1; j++){

#pragma HLS UNROLL
				data_t tmp = 0;
				for(loop_t l=0; l<out; l++){

#pragma HLS LOOP_FLATTEN
					tmp += w2[j*out+l]*d3[l];
				}
				d2[j] = tmp * d_sigmoid(xi2[j]);
			}
			// update the wait of hidden layer
			for(loop_t i=0; i<in+1; i++){

#pragma HLS UNROLL
				for(loop_t j=0; j<hid; j++){

#pragma HLS LOOP_FLATTEN
					w1[i*hid+j] -= eta*d2[j]*o1[i];
				}
			}
		}
}

template<int in, int hid, int out, int sample>
data_t
mlp<in, hid, out, sample>::predict(const data_t *x){
	// forward propagation
	forward(x);
	// biggest output means most probable label
	data_t max = o3[0];
	//int ans = 0;
	data_t ans = 0;
	for(loop_t i=1; i<out; i++){

#pragma HLS UNROLL
		if(o3[i] > max){
			max = o3[i];
			ans = i;
		}
	}
	return ans;
}

//caluculate forward propagation of input x
template<int in, int hid, int out, int sample>
void
mlp<in, hid, out, sample>::forward(const data_t *x){
	//calculation of input layer
	for(loop_t j=0; j<in; j++){

#pragma HLS UNROLL
		xi1[j] = x[j];
		xi1[in] = 1;
		o1[j] = xi1[j];
	}
	o1[in] = 1;

	//calculation of hidden layer
	for(loop_t j=0; j<hid; j++){

#pragma HLS UNROLL
		xi2[j] = 0;
		for(loop_t i=0; i<in+1; i++){

#pragma HLS LOOP_FLATTEN
			xi2[j] += w1[i*hid+j]*o1[i];
		}
		o2[j] = sigmoid(xi2[j]);
	}
	o2[hid] = 1;

	//caluculation of output layer
	for(loop_t j=0; j<out; j++){

#pragma HLS UNROLL
		xi3[j] = 0;
		for(loop_t i=0; i<hid+1; i++){
#pragma HLS LOOP_FLATTEN
			xi3[j] += w2[i*out+j]*o2[i];
		}
		o3[j] = xi3[j];
	}
}

//sigmoid function
template<int in, int hid, int out, int sample>
data_t
mlp<in, hid, out, sample>::sigmoid(data_t x){
	return 1/(1+exp(-x));
	/*
	if (x>0) return x;
	else return 0;
	*/
}

//derivative of sigmoid function
template<int in, int hid, int out, int sample>
data_t
mlp<in, hid, out, sample>::d_sigmoid(data_t x){
	return (1-sigmoid(x))*sigmoid(x);
	/*
	if (x>0) return 1;
	else return 0;
	*/
}

template<int in, int hid, int out, int sample>
void
mlp<in, hid, out, sample>::set_weight(data_t *_w1, data_t *_w2) {
	for(loop8_t i=0; i<(in+1)*hid; i++)

w1[i] = *(_w1+i);
	for(loop8_t i=0; i<(hid+1)*out; i++)

w2[i] = *(_w2+i);
}

template<int in, int hid, int out, int sample>
void
mlp<in, hid, out, sample>::get_weight(data_t *_w1, data_t *_w2) {
	for(loop8_t i=0; i<(in+1)*hid; i++)
		*(_w1+i) = w1[i];
	for(loop8_t i=0; i<(hid+1)*out; i++)
		*(_w2+i) = w2[i];
}
#endif
