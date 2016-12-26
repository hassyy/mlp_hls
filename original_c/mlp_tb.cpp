#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>

#include "mlp_config.h"
#include "mlp_top.h"

using namespace std;

void save(std::string filename, data_t *w1, data_t *w2, int in, int hid, int out) {
	std::ofstream ofs(filename.c_str());
	ofs << "LIBNN_MLP_0" << std::endl;
	ofs << in << " " << hid << " " << out << std::endl;
	for(int i=0; i<(in+1)*hid-1; i++) ofs << w1[i] << " ";
	ofs << w1[(in+1)*hid-1] << std::endl;
	for(int i=0; i<(hid+1)*out-1; i++) ofs << w2[i] << " ";
	ofs << w2[(hid+1)*out-1] << std::endl;
	ofs.close();
}

void load(std::string filename, data_t *w1, data_t *w2, int *in_out, int *hid_out, int *out_out) {
    std::ifstream ifs(filename.c_str());
    if(!ifs){
        std::cout << "File does not exist" << std::endl;
        exit(1);
    }
    std::string str;
    ifs >> str;
    if(str!="LIBNN_MLP_0"){
        std::cout << "File type error!" << std::endl;
        exit(1);
    }
    int in, hid, out;
    ifs >> in >> hid >> out;
    *in_out = in;
    *hid_out = hid;
    *out_out = out;
    float tmp;
    for(int i=0; i<(in+1)*hid; i++) {
    	ifs >> tmp;
    	*(w1+i) = (data_t)tmp;
    }
    for(int i=0; i<(hid+1)*out; i++) {
    	ifs >> tmp;
    	*(w2+i) = (data_t)tmp;
    }
}

int main(void){

	// param file name
	std::string param_file_name = "mlp_train_param.dat";

	// number of test data
	// Use const, or we'll have compile error
	//   due to the possibility of no initialization of w[].
	const int sample = NUM_SAMPLE;
	// size (dimension) of input vector
	const int size = NUM_I_LAYER;  // Fixed
	// number of labels
	const int label = NUM_O_LAYER;
	// number of hidden layer
	const int hid = NUM_H_LAYER;
	// number of training 
	const int repeat = NUM_REPEAT;

	// create MLP(input 2, hidden 3, output 3)
	// number of hidden layer is your choice
	float x_tmp[size*sample];

	// train data
	data_t x[size*sample];
	// label data
	data_t t[sample];
	int tmp_t[sample];
	// Result
	data_t y[sample];

	// Weight 
	data_t 	w1_in[(size+1)*hid] = {};
	data_t	w2_in[(hid+1)*label] = {};
	data_t 	w1_out[(size+1)*hid] = {};
	data_t  w2_out[(hid+1)*label] = {};
	data_t 	w1_dummy[(size+1)*hid] = {};
	data_t  w2_dummy[(hid+1)*label] = {};

	// load CSV
	FILE *fp = fopen("sample.csv", "r");
	if(fp==NULL) return -1;
	for(int i=0; i<sample; i++){
		// load label
		fscanf(fp, "%d,", (tmp_t+i));
		t[i] = (data_t)tmp_t[i];
		// load data
		for(int j=0; j<size; j++) {
			fscanf(fp, "%f,", x_tmp+size*i+j);
			data_t tmp_data = *(x_tmp+size*i+j);
			//printf("*(x_tmp+size*i+j)=%f, tmp_data=%f\n", *(x_tmp+size*i+j), tmp_data.to_float());
			*(x+size*i+j) = tmp_data;
		}
	}

	// Initialize wait by random value
	srand ((unsigned) (time(0)));
	data_t range = std::sqrt(6)/std::sqrt(size+hid+2);
	std::srand ((unsigned) (std::time(0)));
	for(int i=0; i<(size+1)*hid; i++)
		w1_in[i] = (data_t) 2*range*std::rand()/RAND_MAX-range;
	for(int i=0; i<(hid+1)*label; i++)
		w2_in[i] = (data_t) 2*range*std::rand()/RAND_MAX-range;

	// Info.
	cout << "[TB] Test Params." << endl;
	cout << "[TB]   sample:" << sample << endl;
	cout << "[TB]   repeat:" << repeat << endl;
	cout << "[TB] NN Layer." << endl;
	cout << "[TB]   IN_LAYER : " << size << endl;
	cout << "[TB]   HID_LAYER: " << hid << endl;
	cout << "[TB]   OUT_LAYER: " << label << endl << endl;

	// Train
	cout << "[TB] Train ...\n";
	for (int i=0; i<repeat; i++) {
		if (i%100==0) cout << "[TB] train repeat : " << i << endl;
		mlp_top(0, size, sample, w1_in, w2_in, w1_out, w2_out, x, y, t);
		// Copy w_out to w_in for the next train. 
		for(int j=0; j<(size+1)*hid; j++) w1_in[j] = w1_out[j];
		for(int j=0; j<(hid+1)*label; j++) w2_in[j] = w2_out[j];
	}
	cout << "done" << endl;

/*
	// Save train params
	// -> This will be saved under "project_dir/top_name/solution1/csim/build/"
	save(param_file_name, w1_out, w2_out, NUM_I_LAYER, NUM_H_LAYER, NUM_O_LAYER);


	// Load train params
	int l_in, l_hid, l_out;
	load(param_file_name, w1_out, w2_out, &l_in, &l_hid, &l_out);
	if (l_in==size and l_hid==hid and l_out==label)
		cout << "[TB] loaded net size is OK" << endl;
	else {
		cout << "[TB] loaded net size is NG" << endl;
		cout << "l_in=" << l_in << " l_hid=" << l_hid << " l_out=" << l_out << endl;
	}
*/

	// Predict 
	cout << "[TB] Predict ... ";

	for (int i=0; i<sample; i++)
		mlp_top(1, size, sample, w1_out, w2_out, w1_dummy, w2_dummy, x+size*i, y+i, t);
	cout << "done" << endl;


	// show the result
	cout << "[TB] Result of labeling." << endl;
	for(int i=0; i<sample; i++) {
		if (i%10==0) cout << endl;
		cout << y[i] << " ";
	}
	cout << endl;

	cout << "[TB] ALL DONE." << endl;
	return 0;
}


