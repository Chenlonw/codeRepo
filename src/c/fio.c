#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "openfile.h"
#include <sys/types.h>
#include <sys/stat.h>

void read_float_1d(const char *file, float *a, int nx)
/*<write 1D float document>*/
{
	int i;
	FILE *ofile;
	ofile = sfopen(file, "rb");
	for(i=0; i<nx; i++)
		fread(&a[i], sizeof(float), 1, ofile);
	fclose(ofile);
}

void write_float_1d(const char *file, float *a, int nx)
/*<write 1D float document>*/
{
	int i;
	FILE *ofile;
	ofile = sfopen(file, "wb");
	for(i=0; i<nx; i++)
		fwrite(&a[i], sizeof(float), 1, ofile);
	fclose(ofile);
}

void read_float_2d(const char* file, float** a, int nx, int nz)
/*<read 2D float document>*/
{
    int i;
    FILE *ofile;
    ofile=sfopen(file,"rb");
    for(i=0;i<nx;i++)
        fread(&a[i][0],sizeof(float),nz,ofile);
    fclose(ofile);
}

void write_float_2d(const char* file, float** a, int nx, int nz)
/*<write 2D float document>*/
{
	int i;
	FILE *ofile;
	ofile=sfopen(file,"wb");
	for(i=0;i<nx;i++)
		fwrite(&a[i][0],sizeof(float),nz,ofile);
	fclose(ofile);
}

void read_float_3d(const char* file, float*** a, int nx, int ny, int nz)
/*<read 3D float document>*/
{
	int i, j;
	FILE *ofile;
	ofile=sfopen(file,"rb");
	for(i=0;i<ny;i++)
		for(j=0;j<nx;j++)
			fread(&a[i][j][0],sizeof(float),nz,ofile);
	fclose(ofile);
}

void write_float_3d(const char* file, float*** a, int nx, int ny, int nz)
/*<write 3D float document>*/
{
	int i, j;
	FILE *ofile;
	ofile=sfopen(file,"wb");
	for(i=0;i<ny;i++)
		for(j=0;j<nx;j++)
			fwrite(&a[i][j][0],sizeof(float),nz,ofile);
	fclose(ofile);
}

bool file_exists(char *filename) 
/*<Checking if a file with filename exists.>*/
{
	FILE *file;

	if((file = fopen(filename,"r")) != NULL) {
		fclose(file);
		return true;
	}
	
	return false;
}

bool dir_exists(char *dirname)
/*<Checking if a directory with dirname exists.>*/
{
	struct stat s;
	if(stat(dirname,&s) == 0) {
		return true;
	}

	return false;
}

unsigned long count_lines_of_file(const char *file_patch)
/*<Returns number of lines in an ascii file.>*/
{
	FILE *fp = fopen(file_patch, "r");
	unsigned long int n = 0;
	int pc = EOF;
	int c;

	if(fp == NULL){
		fclose(fp);
		return 0;
	}

	while ((c = fgetc(fp)) != EOF) {
		if (c == '\n')
			++n;
		pc = c;
	}
	if (pc != EOF && pc != '\n')
		++n;

	fclose(fp);
	return n;
}

int getnlines(char *filename)
/*<Getting number of lines from file.>*/
{
	FILE *file;
	char c;
	int nlines;

	file = fopen(filename,"r");
	if(file == NULL) {
		fprintf(stderr, "ERROR: File (%s) is not possible to read or do not exist!\n",filename);
	}

	// Looping through all characters in file
	nlines = 0;
	while(1){
		c = fgetc(file);
		if(c == '\n') nlines++;
		if(c == EOF) break;
	}

	// Clearing memory
	fclose(file);

	// Returning number of lines
	return nlines;
}

int num_digit(const int n)
/*<Finding the number of digits in an integer using simple but efficient method.>*/
{
	if (n < 10) return 1;
	if (n < 100) return 2;
	if (n < 1000) return 3;
	if (n < 10000) return 4;
	if (n < 100000) return 5;
	if (n < 1000000) return 6;
	if (n < 10000000) return 7;
	if (n < 100000000) return 8;
	if (n < 1000000000) return 9;
	return 10;
}
