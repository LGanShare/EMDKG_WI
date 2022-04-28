#ifndef SETTING_H
#define SETTING_H
#define INT long
#define REAL float
#include <cstring>
#include <cstdio>
#include <string>

std::string inPath = "../data/";
std::string outPath = "../data/";

extern "C"
void setInPath(char *path){
	INT len = strlen(path);
	inPath = "";
	for(INT i = 0; i < len; i++)
		inPath = inPath + path[i];
	printf("Input files path: %s\n", inPath.c_str());
}

extern "C"
void setOutPath(char *path) {
	INT len = strlen(path);
	outPath = "";
	for (INT i=0; i < len; i++)
		outPath = outPath + path[i];
	printf("Output files path: %s\n", outPath.c_str());
}


INT workThreads = 1;

extern "C"
void setWorkThreads(INT threads) {
	workThreads = threads;
}

extern "C"
INT getWorkThreads() {
	return workThreads;
}

INT userTotal = 0;
INT itemTotal = 0;
INT relationTotal = 0;
INT entityTotal = 0;
INT tripleTotal = 0;
INT testTotal = 0;
INT trainTotal = 0;
INT validTotal = 0;
INT trainUiTotal = 0;

extern "C"
INT getUserTotal() {
	return userTotal;
}

extern "C"
INT getItemTotal() {
	return itemTotal;
}

extern "C"
INT getEntityTotal() {
	return entityTotal;
}

extern "C"
INT getRelationTotal() {
	return relationTotal;
}

extern "C"
INT getTripleTotal() {
	return trainTotal;
}

extern "C"
INT getTrainTotal() {
	return trainTotal;
}

extern "C"
INT getTestTotal() {
	return testTotal;
}

extern "C"
INT getValidTotal() {
	return validTotal;
}

extern "C"
INT getTrainUiTotal() {
	return trainUiTotal;
}

INT bernFlag = 0;

extern "C"
void setBern(INT con) {
	bernFlag = con;
}

#endif

