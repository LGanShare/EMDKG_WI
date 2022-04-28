#ifndef READER_H
#define READER_H
#include "Setting.h"
#include "Triple.h"
#include <cstdlib>
#include <algorithm>

/*
 * freqEnt: stores the frequency of each entity in the training triples
 * freqRel: stores the frequency of each relation in the training triples
 *
 * lefHead: stores the index of each entity as the left of two consecutive heads for trainHead
 * rigHead: stores the index of each entity as the right of two consecutive heads for trainHead
 * lefTail: stores the index of each entity as the left of two consecutive tails for trainTail
 * rigtail: stores the index of each entity as the right of two consecutive tails for trainTail
 * lefRel: stores the index of each entity as the left of two consecutive heads for trainRel
 * rigRel: stores the index of each entity as the right of two consecutive heads for trainRel
 *
 * left_mean: 
 * right_mean: 
 *
 * trainList: stores the triples for training sorted with cmp_head
 * trainHead: stores the triples for training sorted with cmp_head
 * trainTail: stores the triples for training sorted with cmp_tail
 * trainRel: stores triples for training sorted with cmp_rel
 *
 * testLef: stores the starting index of each relation in the testList
 * testRig: stores the last index of each relation in the testList
 * validLef: stores the starting index of each relation in the validList
 * validRig: stores the last index of each relation in the validList
 *
 * testList: stores the triples for testing
 * validList: stores the triples for validation
 * tripleList: stores the triples for training, testing and validation
 *
 * head_lef: stores the starting index of each relation in the head_type
 * head_rig: stores the last index of each relation in the head_type
 * tail_lef: stores the starting index of each relation in the tail_type
 * tail_rig: stores the last index of each relation in the tail_type
 * head_type: stores the total left(head), sorted within the same relation type
 * tail_type: stores the total right(tail) sorted within the same relation type
 */
INT *freqRel, *freqEnt;
INT *lefHead, *rigHead;
INT *lefRel, *rigRel;
INT *lefTail, *rigTail;
INT *lefHeadUi, *rigHeadUi;
INT *lefTailUi, *rigTailUi;
REAL *left_mean, *right_mean;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;
// add user item training triples pointer
Triple *trainUi;

INT *testLef, *testRig;
INT *validLef, *validRig;

extern "C"
void importTrainFiles() {
	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(),"r");
	tmp = fscanf(fin, "%ld", &relationTotal);
	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &entityTotal);
	printf("The total of entities is %ld.\n", entityTotal);
	fclose(fin);
	
	// add reading user and item ids
	fin = fopen((inPath + "user2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &userTotal);
	printf("The total of users is %ld.\n", userTotal);
	fclose(fin);

	fin = fopen((inPath + "item2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &itemTotal);
	printf("The total of items is %ld.\n", itemTotal);
	fclose(fin);	
	
	fin = fopen((inPath + "trainUi2id.txt").c_str(), "r");	
	tmp = fscanf(fin, "%ld", &trainUiTotal);
	printf("The total of training UI triples is %ld.\n", trainUiTotal);
	fclose(fin);

	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &trainTotal);
	printf("The total of training triples is %ld.\n", trainTotal);
	
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	// printf("0");
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));
	// trainUi = (Triple *)calloc(trainUiTotal, sizeof(Triple));
	// printf("1");

	freqRel = (INT *)calloc(relationTotal, sizeof(INT));
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));
	// printf("2");

	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].t);
		tmp = fscanf(fin, "%ld", &trainList[i].r);
		// if (trainList[i].r == 0)
		// 	trainUi[i] = trainList[i];
	}
	fclose(fin);
	printf("Finish reading train files.\n");

	//std::sort(trainList, trainList+trainTotal, Triple::cmp_rel2);
	std::sort(trainList,trainList + trainTotal, Triple::cmp_head);
	
	printf("0");

	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	freqEnt[trainList[0].t] += 1;
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
	
	for (INT i = 1; i < tmp; i++)
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t){
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[i];
			trainTotal++;
			freqEnt[trainList[i].t]++;
			freqEnt[trainList[i].h]++;
			freqRel[trainList[i].r]++;
		}
	// printf("1");

	//std::sort(trainHead,trainHead+trainTotal, Triple::cmp_rel2);
	std::sort(trainHead, trainHead+trainTotal, Triple::cmp_head);
	
	std::sort(trainTail, trainTail+trainTotal, Triple::cmp_tail);
	
	std::sort(trainRel, trainRel+trainTotal, Triple::cmp_rel);

	printf("The total of train triples is %ld.\n", trainTotal);
	
	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	lefRel = (INT *)calloc(entityTotal, sizeof(INT));
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));
	memset(rigHead, -1,sizeof(INT)*(entityTotal));
	memset(rigTail, -1,sizeof(INT)*(entityTotal));
	memset(rigRel, -1,sizeof(INT)*(entityTotal));
	// printf("Writing to lefHead etc..\n");
	for (INT i = 1; i < trainTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			INT index = trainTail[i - 1].t;
			INT index1 = trainTail[i].t;
			rigTail[index] = i - 1;
			lefTail[index1] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			INT index = trainHead[i - 1].h;
			INT index1 = trainHead[i].h;
			rigHead[index] = i - 1;
			lefHead[index1] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h) {
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	// for (INT i = 0; i < 20; i++) {
	// 	printf("lefTail[%ld] = %ld\n", i, lefTail[i]);
	// 	printf("rigTail[%ld] = %ld\n", i, rigTail[i]);
	// }
	
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;
	
	// printf("Writing to lefHeadUi etc.\n");
	// lefHeadUi = (INT *)calloc(userTotal, sizeof(INT));
	// rigHeadUi = (INT *)calloc(userTotal, sizeof(INT));
	// lefTailUi = (INT *)calloc(itemTotal, sizeof(INT));
	// rigTailUi = (INT *)calloc(itemTotal, sizeof(INT));
	// memset(rigHeadUi, -1, sizeof(INT)*userTotal);
	// memset(rigTailUi, -1, sizeof(INT)*itemTotal);
	// printf("Finish writing to lefHeadUi etc.\n");
	// for(INT i = 1; i < trainUiTotal; i++){
		// printf("enter loop..\n");
		// printf("trainHead[%ld]: (%ld,%ld,%ld)\n",i,trainHead[i].h, trainHead[i].t, trainHead[i].r);
	//	if (trainHead[i].h != trainHead[i-1].h){
	//		rigHeadUi[trainHead[i-1].h] = i-1;
	//		lefHeadUi[trainHead[i].h] = i;
	//	}
	// }
	// for (INT i = 1; i < trainUiTotal; i++)
	//	if (trainTail[i].t != trainTail[i-1].t){
	//		rigTailUi[trainTail[i-1].t-userTotal] = i-1;
	//		lefTailUi[trainTail[i].t-userTotal] = i;
	// 	}
	// lefHeadUi[trainHead[0].h] = 0;
	// rigHeadUi[trainHead[trainUiTotal -1].h] = trainUiTotal - 1;
	// lefTailUi[trainTail[0].t-userTotal] = 0;
	// rigTailUi[trainTail[trainUiTotal -1].t-userTotal] = trainUiTotal - 1;

	left_mean = (REAL *)calloc(relationTotal, sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal, sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
	// printf("Finish import train files.\n");
}

Triple *testList;
Triple *validList;
Triple *tripleList;

extern "C"
void importTestFiles() {
	FILE *fin;
	INT tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &entityTotal);
	fclose(fin);

	printf("Import test files...\n");
	FILE* f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");
	FILE* f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
	FILE* f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
	tmp = fscanf(f_kb1, "%ld", &testTotal);
	tmp = fscanf(f_kb2, "%ld", &trainTotal);
	tmp = fscanf(f_kb3, "%ld", &validTotal);
	tripleTotal = testTotal + trainTotal + validTotal;
	testList = (Triple *)calloc(testTotal, sizeof(Triple));
	validList = (Triple *)calloc(validTotal, sizeof(Triple));
	tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));

	// read test2id.txt
	for (INT i = 0; i < testTotal; i++) {
		tmp = fscanf(f_kb1, "%ld", &testList[i].h);
		tmp = fscanf(f_kb1, "%ld", &testList[i].t);
		tmp = fscanf(f_kb1, "%ld", &testList[i].r);
		tripleList[i] = testList[i];
	}

	// read train2id.txt
	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
		tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
		tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
	}

	// read valid2id.txt
	for (INT i = 0; i < validTotal; i++) {
		tmp = fscanf(f_kb3, "%ld", &validList[i].h);
		tmp = fscanf(f_kb3, "%ld", &validList[i].t);
		tmp = fscanf(f_kb3, "%ld", &validList[i].r);
		tripleList[i + testTotal + trainTotal] = validList[i];
	}
	fclose(f_kb1);
	fclose(f_kb2);
	fclose(f_kb3);

	std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
	std::sort(testList, testList + testTotal, Triple::cmp_rel2);
	std::sort(validList, validList + validTotal, Triple::cmp_rel2);
	printf("The total of test triples is %ld.\n", testTotal);
	printf("The total of valid triples is %ld.\n",  validTotal);

	testLef = (INT *)calloc(relationTotal, sizeof(INT));
	testRig = (INT *)calloc(relationTotal, sizeof(INT));
	memset(testLef, -1, sizeof(INT) * relationTotal);
	memset(testRig, -1, sizeof(INT) * relationTotal);
	for (INT i = 1; i < testTotal; i++) {
		if (testList[i].r != testList[i-1].r) {
			testRig[testList[i-1].r] = i - 1;
			testLef[testList[i].r] = i;
		}
	}
	testLef[testList[0].r] = 0;
	testRig[testList[testTotal - 1].r] = testTotal - 1;

	validLef = (INT *)calloc(relationTotal, sizeof(INT));
	validRig = (INT *)calloc(relationTotal, sizeof(INT));
	memset(validLef, -1, sizeof(INT)*relationTotal);
	memset(validRig, -1, sizeof(INT)*relationTotal);
	for (INT i = 1; i < validTotal; i++) {
		if (validList[i].r != validList[i-1].r) {
			validRig[validList[i-1].r] = i - 1;
			validLef[validList[i].r] = i;
		}
	}
	validLef[validList[0].r] = 0;
	validRig[validList[validTotal - 1].r] = validTotal - 1;
}


INT* head_lef;
INT* head_rig;
INT* tail_lef;
INT* tail_rig;
INT* head_type;
INT* tail_type;

extern "C"
void importTypeFiles() {
	head_lef = (INT *)calloc(relationTotal, sizeof(INT));
	head_rig = (INT *)calloc(relationTotal, sizeof(INT));
	tail_lef = (INT *)calloc(relationTotal, sizeof(INT));
	tail_rig = (INT *)calloc(relationTotal, sizeof(INT));
	INT total_lef = 0;
	INT total_rig = 0;
	printf("Import type constraint file.\n");
	FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(), "r");
	INT tmp;
	tmp = fscanf(f_type, "%ld", &tmp);
	// printf("importTypeFiles sign 1.\n");
	for (INT i = 0; i < relationTotal; i++) {
		// printf("Enter for 1\n");
		INT rel, tot;
		tmp = fscanf(f_type, "%ld %ld", &rel, &tot);
		// printf("tot: %ld\n", tot);
		for (INT j = 0; j < tot; j++) {
			// printf("Enter for 1.1.\n");
			tmp = fscanf(f_type, "%ld", &tmp);
			total_lef++;
		}
		// printf("Leave for 1.1.\n");
		tmp = fscanf(f_type, "%ld%ld", &rel,&tot);
		// printf("tot: %ld\n", tot);
		for (INT j = 0; j < tot; j++) {
			// if (j == 0)printf("Enter for 1.2.\n");
			tmp = fscanf(f_type, "%ld", &tmp);
			total_rig++;
		}
		// printf("Leave for 1.2.\n\n");
	}
	fclose(f_type);
	head_type = (INT *)calloc(total_lef, sizeof(INT));
	tail_type = (INT *)calloc(total_rig, sizeof(INT));
	total_lef = 0;
	total_rig = 0;
	f_type = fopen((inPath + "type_constrain.txt").c_str(), "r");
	tmp = fscanf(f_type, "%ld", &tmp);
	for (INT i = 0; i < relationTotal; i++) {
		printf("Enter for 2.\n");
		INT rel, tot;
		tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
		head_lef[rel] = total_lef;
		printf("head_lef[%ld] = %ld\n",rel,total_lef);
		for (INT j = 0; i < tot; i++) {
			tmp = fscanf(f_type, "%ld", &head_type[total_lef]);
			total_lef++;
		}
		head_rig[rel] = total_lef;
		std::sort(head_type + head_lef[rel],head_type + head_rig[rel]);
		tmp = fscanf(f_type,"%ld%ld", &rel,&tot);
		tail_lef[rel] = total_rig;
		for (INT j = 0; j < tot; j++) {
			tmp = fscanf(f_type, "%ld", &tail_type[total_rig]);
			total_rig++;
		}
		tail_rig[rel] = total_rig;
		std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
	}
	fclose(f_type);
	printf("Finish import type constraint file.\n");
}

#endif
