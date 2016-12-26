#include "mlp_top.h"
#include "mlp.h"


void mlp_top (
	const int mode
	, const int size
	, const int sample
	, data_t *w1_in
	, data_t *w2_in
	, data_t *w1_out
	, data_t *w2_out
	, const data_t *x
	, data_t *y
	, const data_t *t
) {
#pragma HLS DATAFLOW
#pragma HLS INTERFACE axis depth=300 port=t
#pragma HLS INTERFACE axis depth=300 port=y
#pragma HLS INTERFACE axis depth=300 port=x
#pragma HLS INTERFACE axis depth=12 port=w2_out
#pragma HLS INTERFACE axis depth=9 port=w1_out
#pragma HLS INTERFACE axis depth=12 port=w2_in
#pragma HLS INTERFACE axis depth=9 port=w1_in
#pragma HLS INTERFACE s_axilite depth=1 port=sample offset=8 bundle=REG_BUS
#pragma HLS INTERFACE s_axilite depth=1 port=size offset=4 bundle=REG_BUS
#pragma HLS INTERFACE s_axilite depth=1 port=mode offset=0 bundle=REG_BUS
#pragma HLS INTERFACE s_axilite depth=1 port=return bundle=REG_BUS

	//static
	mlp< NUM_I_LAYER
		, NUM_H_LAYER
		, NUM_O_LAYER
		//, NUM_REPEAT
		, NUM_SAMPLE
		>
		net;

	if (mode==0) {
		net.set_weight(w1_in, w2_in);
		net.train(x, t, 0.1);
		net.get_weight(w1_out, w2_out);
	} else if (mode==1) {
		net.set_weight(w1_in, w2_in);
		*y = net.predict(x);
	}/* else if (mode==2) {
		net.set_weight(w1_in, w2_in);
	} else {
		net.get_weight(w1_out, w2_out);
	}*/
}
