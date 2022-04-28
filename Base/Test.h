#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"
#include <stdio.h>
#include <cstdlib>
#include <math.h>

// fptr, fptr_1, fptr_2 are used to manipulate result files
FILE *fptr;
FILE *fptr_1;
FILE *fptr_2;

//
INT lastHead = 0;
INT lastTail = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;

REAL l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
REAL l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;

// new variables for NDCG metrics
REAL l_filter_b_DCG = 0, l_filter_b_DCG_constrain = 0, l_b_DCG = 0, l_b_DCG_constrain = 0;
REAL r_filter_b_DCG = 0, r_filter_b_DCG_constrain = 0, r_b_DCG = 0, r_b_DCG_constrain = 0;

REAL iDCG = 1.0;

extern "C"
void initTest() {
    lastHead = 0;
    lastTail = 0;
    l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
    l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;

    l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
    l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
}

// getHeadBatch: to get all possible (h, t, r) given a fixed t and r.
extern "C"
void getHeadBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < userTotal; i++) {
        ph[i] = i;
        pt[i] = testList[lastHead].t;
        pr[i] = testList[lastHead].r;
    }
    // printf("Get head batch for %ld\n",lastHead);
}

// getTailBatch: to get all possible (h, t, r) given a fixed h and r.
extern "C"
void getTailBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = userTotal; i < userTotal + itemTotal; i++) {
        ph[i] = testList[lastTail].h;
        pt[i] = i;
        pr[i] = testList[lastTail].r;
    }
}


extern "C"
void testHead(REAL *con){
    // printf("c++ sign 0.\n");
    INT h = testList[lastHead].h;
    INT t = testList[lastHead].t;
    INT r = testList[lastHead].r;
    // printf("c++ sign 0.1. \n");
    INT lef = head_lef[r], rig = head_rig[r];
    // printf("c++ sign 0.2. \n");
    REAL minimal = con[h];
    // printf("c++ sign 1.\n");
    // l_s is the rank among the headBatch results
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;
    INT l_filter_s_constrain = 0;
    // printf("c++ sign 2.\n");
    INT count = 0;
    for (INT j = 0; j < userTotal; j++) {
	    // count += 1;
	    if (j != h) {
	    // count += 1;
	    REAL value = con[j];
            if (value < minimal) {
                l_s += 1;
                if (not _find(j, t, r))
                    l_filter_s += 1;
            }
            while (lef < rig && head_type[lef] < j) lef ++;
            if (lef < rig && j == head_type[lef]) {
                if (value < minimal) {
                    l_s_constrain += 1;
                    if (not _find(j, t, r)) {
                        l_filter_s_constrain += 1;
                    }
                }
            }
        }
    }

    if (l_filter_s < 10) l_filter_tot += 1;
    if (l_s < 10) l_tot += 1;

    // calculate DCG@10
    if (l_filter_s < 10)l_filter_b_DCG += log(2.0)/log(l_filter_s+2.0);
    if (l_s < 10)l_b_DCG += log(2)/log(l_s+2.0);

    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 1) l1_tot += 1;

    if (l_filter_s_constrain < 10) l_filter_tot_constrain += 1;
    if (l_s_constrain < 10) l_tot_constrain += 1;

    // calculate DCG@10
    if (l_filter_s_constrain < 10)l_filter_b_DCG_constrain += log(2)/log(l_filter_s_constrain+2.0);
    if (l_s_constrain < 10)l_b_DCG_constrain += log(2)/log(l_s_constrain+2.0);

    if (l_filter_s_constrain < 3) l3_filter_tot_constrain += 1;
    if (l_s_constrain < 3) l3_tot_constrain += 1;
    if (l_filter_s_constrain < 1) l1_filter_tot_constrain += 1;
    if (l_s_constrain < 1) l1_tot_constrain += 1;

    l_filter_rank += (l_filter_s+1);
    l_rank += (1+l_s);
    l_filter_reci_rank += 1.0/(l_filter_s+1);
    l_reci_rank += 1.0/(l_s+1);

    l_filter_rank_constrain += (l_filter_s_constrain+1);
    l_rank_constrain += (1+l_s_constrain);
    l_filter_reci_rank_constrain += 1.0/(l_filter_s_constrain+1);
    l_reci_rank_constrain += 1.0/(l_s_constrain+1);

    lastHead++;
    printf("Finish testHead.");
}


extern "C"
void testTail(REAL *con) {
    INT h = testList[lastTail].h;
    INT t = testList[lastTail].t;
    INT r = testList[lastTail].r;
    INT lef = tail_lef[r], rig = tail_rig[r];
    REAL minimal = con[t];
    INT r_s = 0;
    INT r_filter_s = 0;
    INT r_s_constrain = 0;
    INT r_filter_s_constrain = 0;
    INT count = 0;
    for (INT j = userTotal; j < (userTotal + itemTotal); j++) {
	// count += 1;
	    if (j != t) {
            // count += 1;
	    REAL value = con[j];
	    // j = j + userTotal;
            if (value < minimal) {
                r_s += 1;
                if (not _find(h, j, r))
                    r_filter_s += 1;
            }
            while (lef < rig && tail_type[lef] < j) lef ++;
            if (lef < rig && j == tail_type[lef]) {
                    if (value < minimal) {
                        r_s_constrain += 1;
                        if (not _find(h, j ,r)) {
                            r_filter_s_constrain += 1;
                        }
                    }
            }
        }

    }

    if (r_filter_s < 10) r_filter_tot += 1;
    if (r_s < 10) r_tot += 1;

     // calculate DCG@10
    if (r_filter_s < 10)r_filter_b_DCG += log(2)/log(r_filter_s+2.0);
    if (r_s < 10)l_b_DCG += log(2)/log(r_s+2.0);

    if (r_filter_s < 3) r3_filter_tot += 1;
    if (r_s < 3) r3_tot += 1;
    if (r_filter_s < 1) r1_filter_tot += 1;
    if (r_s < 1) r1_tot += 1;

    if (r_filter_s_constrain < 10) r_filter_tot_constrain += 1;
    if (r_s_constrain < 10) r_tot_constrain += 1;

    // calculate DCG@10
    if (r_filter_s_constrain < 10)r_filter_b_DCG_constrain += log(2)/log(r_filter_s_constrain+2.0);
    if (r_s_constrain < 10)r_b_DCG_constrain += log(2)/log(r_s_constrain+2.0);

    if (r_filter_s_constrain < 3) r3_filter_tot_constrain += 1;
    if (r_s_constrain < 3) r3_tot_constrain += 1;
    if (r_filter_s_constrain < 1) r1_filter_tot_constrain += 1;
    if (r_s_constrain < 1) r1_tot_constrain += 1;

    r_filter_rank += (1+r_filter_s);
    r_rank += (1+r_s);
    r_filter_reci_rank += 1.0/(1+r_filter_s);
    r_reci_rank += 1.0/(1+r_s);

    r_filter_rank_constrain += (1+r_filter_s_constrain);
    r_rank_constrain += (1+r_s_constrain);
    r_filter_reci_rank_constrain += 1.0/(1+r_filter_s_constrain);
    r_reci_rank_constrain += 1.0/(1+r_s_constrain);

    lastTail++;
}

extern "C"
void test_recommendation() {
    l_rank /= testTotal;
    r_rank /= testTotal;
    l_reci_rank /= testTotal;
    r_reci_rank /= testTotal;

    l_tot /= testTotal;
    l3_tot /= testTotal;
    l1_tot /= testTotal;

    r_tot /= testTotal;
    r3_tot /= testTotal;
    r1_tot /= testTotal;

    // with filter
    l_filter_rank /= testTotal;
    r_filter_rank /= testTotal;
    l_filter_reci_rank /= testTotal;
    r_filter_reci_rank /= testTotal;

    l_filter_tot /= testTotal;
    l3_filter_tot /= testTotal;
    l1_filter_tot /= testTotal;

    r_filter_tot /= testTotal;
    r3_filter_tot /= testTotal;
    r1_filter_tot /= testTotal;

    // for (INT j=2; j < testTotal; j++)
    // {
    //   iDCG +=log(2.0)/log(j+1.0);
    // }

    // calculte NDCG@10 for heads
    l_filter_b_DCG /= testTotal;
    l_b_DCG /= testTotal;
    l_filter_b_DCG_constrain /= testTotal;
    l_b_DCG_constrain /= testTotal;

    // calculate NDCG@10 for tails
    r_filter_b_DCG /= testTotal;
    r_b_DCG /= testTotal;
    r_filter_b_DCG_constrain /= testTotal;
    r_b_DCG_constrain /= testTotal;

    printf("no type constraint results:\n");

    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \t NDCG@10\n");
    printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \t %f\n", l_reci_rank, l_rank, l_tot, l3_tot, l1_tot, iDCG);
    printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \t %f\n", r_reci_rank, r_rank, r_tot, r3_tot, r1_tot, r_b_DCG);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \t %f\n",
            (l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l_tot+r_tot)/2, (l3_tot+r3_tot)/2, (l1_tot+r1_tot)/2, (l_b_DCG+r_b_DCG)/2);
    printf("\n");
    printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \t %f\n", l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot, l_filter_b_DCG);
    printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \t %f\n", r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot, r_filter_b_DCG);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \t %f\n",
            (l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l_filter_tot+r_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l1_filter_tot+r1_filter_tot)/2, (l_filter_b_DCG+r_filter_b_DCG)/2);

    // Output results to csv file
    //std::ofstream no_constraint_file;
    //no_constraint_file.open('../results/no_constraint_test_link.csv')
    // outPath
    printf("Start outputing to file ... \n");
    //fptr = fopen("/home/lugan/Documents/cplusplus/OpenKE/results/no_constraint_test_link.txt", "w");
    fptr = fopen((outPath + "no_constraint_test_link.txt").c_str(), "w");

    if(fptr == NULL){
      perror("Could not create file no_constraint_test_link.txt ");
      return;
    }
    fprintf(fptr, "metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n" );
    fprintf(fptr, "l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank, l_rank, l_tot, l3_tot, l1_tot);
    fprintf(fptr, "r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank, r_rank, r_tot, r3_tot, r1_tot);
    fprintf(fptr, "averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            (l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l_tot+r_tot)/2, (l3_tot+r3_tot)/2, (l1_tot+r1_tot)/2);
    fprintf(fptr, "l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot);
    fprintf(fptr, "r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot);
    fprintf(fptr, "averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            (l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l_filter_tot+r_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l1_filter_tot+r1_filter_tot)/2);
    fclose(fptr);

    //type constrain
    l_rank_constrain /= testTotal;
    r_rank_constrain /= testTotal;
    l_reci_rank_constrain /= testTotal;
    r_reci_rank_constrain /= testTotal;

    l_tot_constrain /= testTotal;
    l3_tot_constrain /= testTotal;
    l1_tot_constrain /= testTotal;

    r_tot_constrain /= testTotal;
    r3_tot_constrain /= testTotal;
    r1_tot_constrain /= testTotal;

    // with filter
    l_filter_rank_constrain /= testTotal;
    r_filter_rank_constrain /= testTotal;
    l_filter_reci_rank_constrain /= testTotal;
    r_filter_reci_rank_constrain /= testTotal;

    l_filter_tot_constrain /= testTotal;
    l3_filter_tot_constrain /= testTotal;
    l1_filter_tot_constrain /= testTotal;

    r_filter_tot_constrain /= testTotal;
    r3_filter_tot_constrain /= testTotal;
    r1_filter_tot_constrain /= testTotal;

    printf("type constraint results:\n");

    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \t NDCG@10\n");
    printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \t %f\n", l_reci_rank_constrain, l_rank_constrain, l_tot_constrain, l3_tot_constrain, l1_tot_constrain, l_b_DCG_constrain);
    printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \t %f\n", r_reci_rank_constrain, r_rank_constrain, r_tot_constrain, r3_tot_constrain, r1_tot_constrain, r_b_DCG_constrain);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \t %f\n",
            (l_reci_rank_constrain+r_reci_rank_constrain)/2, (l_rank_constrain+r_rank_constrain)/2, (l_tot_constrain+r_tot_constrain)/2, (l3_tot_constrain+r3_tot_constrain)/2, (l1_tot_constrain+r1_tot_constrain)/2, (l_b_DCG_constrain+r_b_DCG_constrain)/2);
    printf("\n");
    printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \t %f\n", l_filter_reci_rank_constrain, l_filter_rank_constrain, l_filter_tot_constrain, l3_filter_tot_constrain, l1_filter_tot_constrain, l_filter_b_DCG_constrain);
    printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \t %f\n", r_filter_reci_rank_constrain, r_filter_rank_constrain, r_filter_tot_constrain, r3_filter_tot_constrain, r1_filter_tot_constrain, r_filter_b_DCG_constrain);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \t %f\n",
            (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2, (l_filter_rank_constrain+r_filter_rank_constrain)/2, (l_filter_tot_constrain+r_filter_tot_constrain)/2, (l3_filter_tot_constrain+r3_filter_tot_constrain)/2, (l1_filter_tot_constrain+r1_filter_tot_constrain)/2, (l_filter_b_DCG_constrain+r_filter_b_DCG_constrain)/2);

    printf("Start outputing to file ... \n");
    // Output results to csv file
    // fptr_1 = fopen("/home/lugan/Documents/cplusplus/OpenKE/results/type_constraint_test_link.txt", "w");
    fptr_1 = fopen((outPath + "type_constraint_test_link.txt").c_str(), "w");
    if(fptr_1 == NULL){
      //fprintf(stderr, "Value of errno: %d\n", errno);
      perror("Could not create file type_constraint_test_link.txt ");
      return;
    }
    fprintf(fptr_1, "metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n" );
    //fprintf(fptr_1, "metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    fprintf(fptr_1, "l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank_constrain, l_rank_constrain, l_tot_constrain, l3_tot_constrain, l1_tot_constrain);
    fprintf(fptr_1, "r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank_constrain, r_rank_constrain, r_tot_constrain, r3_tot_constrain, r1_tot_constrain);
    fprintf(fptr_1, "averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            (l_reci_rank_constrain+r_reci_rank_constrain)/2, (l_rank_constrain+r_rank_constrain)/2, (l_tot_constrain+r_tot_constrain)/2, (l3_tot_constrain+r3_tot_constrain)/2, (l1_tot_constrain+r1_tot_constrain)/2);
    //printf("\n");
    fprintf(fptr_1, "l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank_constrain, l_filter_rank_constrain, l_filter_tot_constrain, l3_filter_tot_constrain, l1_filter_tot_constrain);
    fprintf(fptr_1, "r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank_constrain, r_filter_rank_constrain, r_filter_tot_constrain, r3_filter_tot_constrain, r1_filter_tot_constrain);
    fprintf(fptr_1, "averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2, (l_filter_rank_constrain+r_filter_rank_constrain)/2, (l_filter_tot_constrain+r_filter_tot_constrain)/2, (l3_filter_tot_constrain+r3_filter_tot_constrain)/2, (l1_filter_tot_constrain+r1_filter_tot_constrain)/2);
    fclose(fptr_1);
    printf("Finish outputing to file ...\n");
}


/*=====================================================================================
triple classification
======================================================================================*/
Triple *negTestList;
extern "C"
void getNegTest() {
    negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        negTestList[i] = testList[i];
        negTestList[i].t = corrupt(testList[i].h, testList[i].r);
    }
    /*
    FILE* fout = fopen((inPath + "test_neg.txt").c_str(), "w");
    for (INT i = 0; i < testTotal; i++) {
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", testList[i].h, testList[i].t, testList[i].r, INT(1));
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", negTestList[i].h, negTestList[i].t, negTestList[i].r, INT(-1));
    }
    fclose(fout);
    */
}

Triple *negValidList;
extern "C"
void getNegValid() {
    negValidList = (Triple *)calloc(validTotal, sizeof(Triple));
    for (INT i = 0; i < validTotal; i++) {
        negValidList[i] = validList[i];
        negValidList[i].t = corrupt(validList[i].h, validList[i].r);
    }
    /*
    FILE* fout = fopen((inPath + "valid_neg.txt").c_str(), "w");
    for (INT i = 0; i < validTotal; i++) {
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", validList[i].h, validList[i].t, validList[i].r, INT(1));
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", negValidList[i].h, negValidList[i].t, negValidList[i].r, INT(-1));
    }
    fclose(fout);
    */
}

extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        ph[i] = testList[i].h;
        pt[i] = testList[i].t;
        pr[i] = testList[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}

extern "C"
void getValidBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegValid();
    for (INT i = 0; i < validTotal; i++) {
        ph[i] = validList[i].h;
        pt[i] = validList[i].t;
        pr[i] = validList[i].r;
        nh[i] = negValidList[i].h;
        nt[i] = negValidList[i].t;
        nr[i] = negValidList[i].r;
    }
}
REAL threshEntire;
extern "C"
void getBestThreshold(REAL *relThresh, REAL *score_pos, REAL *score_neg) {
    REAL interval = 0.01;
    REAL min_score, max_score, bestThresh, tmpThresh, bestAcc, tmpAcc;
    INT n_interval, correct, total;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1) continue;
        total = (validRig[r] - validLef[r] + 1) * 2;
        min_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] < min_score) min_score = score_neg[validLef[r]];
        max_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] > max_score) max_score = score_neg[validLef[r]];
        for (INT i = validLef[r]+1; i <= validRig[r]; i++) {
            if(score_pos[i] < min_score) min_score = score_pos[i];
            if(score_pos[i] > max_score) max_score = score_pos[i];
            if(score_neg[i] < min_score) min_score = score_neg[i];
            if(score_neg[i] > max_score) max_score = score_neg[i];
        }
        n_interval = INT((max_score - min_score)/interval);
        for (INT i = 0; i <= n_interval; i++) {
            tmpThresh = min_score + i * interval;
            correct = 0;
            for (INT j = validLef[r]; j <= validRig[r]; j++) {
                if (score_pos[j] <= tmpThresh) correct ++;
                if (score_neg[j] > tmpThresh) correct ++;
            }
            tmpAcc = 1.0 * correct / total;
            if (i == 0) {
                bestThresh = tmpThresh;
                bestAcc = tmpAcc;
            } else if (tmpAcc > bestAcc) {
                bestAcc = tmpAcc;
                bestThresh = tmpThresh;
            }
        }
        relThresh[r] = bestThresh;
    }
}

REAL *testAcc;
REAL aveAcc;
extern "C"
void test_triple_classification(REAL *relThresh, REAL *score_pos, REAL *score_neg) {
    testAcc = (REAL *)calloc(relationTotal, sizeof(REAL));
    INT aveCorrect = 0, aveTotal = 0;
    REAL aveAcc;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1 || testLef[r] ==-1) continue;
        INT correct = 0, total = 0;
        for (INT i = testLef[r]; i <= testRig[r]; i++) {
            if (score_pos[i] <= relThresh[r]) correct++;
            if (score_neg[i] > relThresh[r]) correct++;
            total += 2;
        }
        testAcc[r] = 1.0 * correct / total;
        aveCorrect += correct;
        aveTotal += total;
    }
    aveAcc = 1.0 * aveCorrect / aveTotal;
    printf("triple classification accuracy is %lf\n", aveAcc);


    //fptr_2 = fopen("/home/lugan/Documents/cplusplus/OpenKE/results/test_triple.txt", "w");
    fptr_2 = fopen((outPath + "test_triple.txt").c_str(), "w");
    fprintf(fptr_2, "triple classification accuracy is %lf\n", aveAcc);
    fclose(fptr_2); 
}


#endif
