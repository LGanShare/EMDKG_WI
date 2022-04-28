#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include "Valid.h"
#include <cstdlib>
#include <pthread.h>

extern "C"
void setInPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

struct Parameter{
	INT id;
	// add batch_u, batch_i
	// INT *batch_u;
	// INT *batch_i;
	INT *batch_h;
	INT *batch_t;
	INT *batch_r;
	REAL *batch_y;
	// add batchSizeUi
	// INT batch_size_ui;
	INT batch_size;
	INT negRate;
	INT negRelRate;
};

void* getBatch(void* con){
	// printf("start getting one batch..\n");
	Parameter *para = (Parameter*)(con);
	INT id = para -> id;

	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;

	INT batch_size = para -> batch_size;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	INT lef, rig; // lef2, rig2;

	// add batch_size_er for the rest entity relations
	// INT batch_size_er = batch_size - batch_size_ui;
	// printf("batch size: %ld.\n", batch_size);
	// printf("batch size UI: %ld.\n", batch_size_ui);
	// printf("batch size ER: %ld.\n", batch_size_er);

	if (batch_size % workThreads == 0){
		lef = id * (batch_size / workThreads);
		rig = lef + (batch_size / workThreads);
	}else{
		lef = id * (batch_size / workThreads + 1);
		rig = lef + (batch_size / workThreads + 1);
		if (rig > batch_size) rig = batch_size;
	}
	// printf("lef1: %ld, rig1: %ld\n", lef1, rig1);

	// if (batch_size % workThreads == 0){
	// 	lef2 = rig1;
	// 	rig2 = rig1 + (batch_size_er / workThreads);
	// }else{
	// 	lef2 = rig1;
	// 	rig2 = (id + 1) * (batch_size_ui / workThreads + 1);
	// 	if (rig2 > batch_size) rig2 = batch_size;
	// }
	// printf("lef2: %ld, rig2: %ld\n", lef2, rig2);

	REAL prob = 500;

	// printf("select for batch_er.\n");
	// choose batch for entity relation triples
	for (INT batch = lef; batch < rig; batch++) {
		INT i = rand_max(id, trainTotal);
		batch_h[batch] = trainList[i].h;
		batch_t[batch] = trainList[i].t;
		batch_r[batch] = trainList[i].r;
		// printf("ER: (%ld, %ld, %ld)\n",trainList[i].h, trainList[i].t, trainList[i].r);
		batch_y[batch] = 1;
		INT last = batch_size;
		for (INT times = 0; times < negRate; times++) {
			if (bernFlag)
				prob = 1000 * right_mean[trainList[i].r] /(right_mean[trainList[i].r] + left_mean[trainList[i].r]);
			if (randd(id) % 1000 < prob) {
				batch_h[batch + last] = trainList[i].h;
				batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
				batch_r[batch + last] = trainList[i].r;
				// printf("batch left %ld: (%ld, %ld, %ld)\n", batch, batch_h[batch+last], batch_t[batch+last], batch_r[batch+last]);
			}else{
				batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
				batch_t[batch + last] = trainList[i].t;
				batch_r[batch + last] = trainList[i].r;
				// printf("batch right %ld: (%ld, %ld, %ld)\n", batch, batch_h[batch+last], batch_t[batch+last], batch_r[batch+last]);
			}
			batch_y[batch + last] = -1;
			last += batch_size;
		}
		for (INT times = 0; times < negRelRate; times++) {
			batch_h[batch + last] = trainList[i].h;
			batch_t[batch + last] = trainList[i].t;
			batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t);
			batch_y[batch + last] = -1;
			last += batch_size;
		}
	}

	// printf("Select for batch_ui.\n");
	// choose triples for  ui parts
	// corrupt_head for ui, corrupt_tail for ui
	// for (INT batch = lef1; batch < rig1; batch++) {
	// 	INT i = rand_max(id, trainUiTotal);
	// 	batch_u[batch] = trainList[i].h;
	//	batch_i[batch] = trainList[i].t - userTotal;
	//	batch_r[batch] = trainList[i].r;
	//	batch_y[batch] = 1;
	//	INT last = batch_size;
	//	for (INT times = 0; times < negRate; times++) {
	//		if (bernFlag)
	//			prob = 1000 * right_mean[trainList[i].r] /(right_mean[trainList[i].r] + left_mean[trainList[i].r]);
	//		if (randd(id) % 1000 < prob) {
	//			batch_u[batch + last] = trainList[i].h;
	//			batch_i[batch + last] = corrupt_head_ui(id, trainList[i].h, trainList[i].r);
	//			batch_r[batch + last] = trainList[i].r;
	//		}else{
	//			batch_u[batch + last] = corrupt_tail_ui(id, trainList[i].t, trainList[i].r);
	//			batch_i[batch + last] = trainList[i].t - userTotal;
	//			batch_r[batch + last] = trainList[i].r;
	//		}
	//		batch_y[batch + last] = -1;
	//		last += batch_size;
	//	}
	//	for (INT times = 0; times < negRelRate; times++) {
	//		batch_u[batch + last] = trainList[i].h;
	//		batch_i[batch + last] = trainList[i].t;
	//		batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t);
	//		batch_y[batch + last] = -1;
	//		last += batch_size;
	//	}
	//}
	
	pthread_exit(NULL);
}

extern "C"
void sampling(INT *batch_h, INT *batch_t, INT *batch_r, REAL *batch_y,INT batchSize, INT negRate = 1, INT negRelRate = 0) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads *sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;

		para[threads].batch_h = batch_h;
		para[threads].batch_t = batch_t;
		para[threads].batch_r = batch_r;
		para[threads].batch_y = batch_y;
		
		// para[threads].batch_size_ui = batchSizeUi;

		para[threads].batch_size = batchSize;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++) 
		pthread_join(pt[threads], NULL);
	free(pt);
	free(para);
	// printf("Safely leave sampling in C++.\n");
}

int main() {
	importTrainFiles();
	return 0;
}
