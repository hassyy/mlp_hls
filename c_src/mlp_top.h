#ifndef __MLP_TOP_H__
#define __MLP_TOP_H__

#include "mlp_config.h"

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
);

#endif
