/* This file is automatically generated. DO NOT EDIT! */

#ifndef _chlwang_cuda_h
#define _chlwang_cuda_h

void *alloc1 (size_t n1, size_t size);
/*< allocate a 1-d array >*/


void free1 (void *p);
/*< free a 1-d array >*/


void **alloc2 (size_t n1, size_t n2, size_t size);
/*< allocate a 2-d array >*/


void free2 (void **p);
/*< free a 2-d array >*/


void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size);
/*< allocate a 3-d array >*/


void free3 (void ***p);
/*< free a 3-d array >*/


//===================1D=====================
int *alloc_int_1d(size_t n1);
/*< allocate a 1-d array of ints >*/


float *alloc_float_1d(size_t n1);
/*< allocate a 1-d array of floats >*/


double *alloc_double_1d(size_t n1);
/*< allocate a 1-d array of double >*/


void free_int_1d(int *p);
/*< free a 1-d array of ints >*/


void free_float_1d(float *p);
/*< free a 1-d array of floats >*/


void free_double_1d(double *p);
/*< free a 1-d array of floats >*/


void zero_int_1d(int *a, int n1);
/*<zero 1D int document>*/


void zero_float_1d(float *a, int n1);
/*<zero 1D float document>*/


void zero_double_1d(double *a, int n1);
/*<zero 1D double document>*/


//===================2D=====================
/* n1: fast dimension; n2: slow dimension */
int **alloc_int_2d(size_t n1, size_t n2);
/*< allocate a 2-d array of ints >*/


float **alloc_float_2d(size_t n1, size_t n2);
/*< allocate a 2-d array of float >*/


double **alloc_double_2d(size_t n1, size_t n2);
/*< allocate a 2-d array of double >*/


void free_int_2d(int **p);
/*< free a 2-d array of ints >*/


void free_float_2d(float **p);
/*< free a 2-d array of floats >*/


void free_double_2d(double **p);
/*< free a 2-d array of floats >*/


void zero_int_2d(int **a, int n1, int n2);
/*<zero 2D int document n1 is the fast axis>*/


void zero_float_2d(float **a, int n1, int n2);
/*<zero 2D float document n1 is the fast axis>*/


void zero_double_2d(double **a, int n1, int n2);
/*<zero 2D double document n1 is the fast axis>*/


//===================3D=====================
int ***alloc_int_3d(size_t n1, size_t n2, size_t n3);
/*< allocate a 3-d array of ints >*/


float ***alloc_float_3d(size_t n1, size_t n2, size_t n3);
/*< allocate a 3-d array of floats >*/


double ***alloc_double_3d(size_t n1, size_t n2, size_t n3);
/*< allocate a 3-d array of doubles >*/


void free_int_3d(int ***p);
/*< free a 3-d array of ints >*/


void free_float_3d(float ***p);
/*< free a 3-d array of floats >*/


void free_double_3d(double ***p);
/*< free a 3-d array of doubles >*/


void zero_int_3d(int ***a, int n1, int n2, int n3);
/*<zero 2D int document n1 is the fast axis>*/


void zero_float_3d(float ***a, int n1, int n2, int n3);
/*<zero 2D float document n1 is the fast axis>*/


void zero_double_3d(double ***a, int n1, int n2, int n3);
/*<zero 2D double document n1 is the fast axis>*/


FILE *sfopen(const char *fn, const char *stat);
/*<safely open the files>*/


void read_float_1d(const char *file, float *a, int nx);
/*<write 1D float document>*/


void write_float_1d(const char *file, float *a, int nx);
/*<write 1D float document>*/


void read_float_2d(const char* file, float** a, int nx, int nz);
/*<read 2D float document>*/


void write_float_2d(const char* file, float** a, int nx, int nz);
/*<write 2D float document>*/


void read_float_3d(const char* file, float*** a, int nx, int ny, int nz);
/*<read 3D float document>*/


void write_float_3d(const char* file, float*** a, int nx, int ny, int nz);
/*<write 3D float document>*/


bool file_exists(char *filename);
/*<Checking if a file with filename exists.>*/


bool dir_exists(char *dirname);
/*<Checking if a directory with dirname exists.>*/


unsigned long count_lines_of_file(const char *file_patch);
/*<Returns number of lines in an ascii file.>*/


int getnlines(char *filename);
/*<Getting number of lines from file.>*/


int num_digit(const int n);
/*<Finding the number of digits in an integer using simple but efficient method.>*/


void utils_print_title(const char *title);
/*<Printing out program title to terminal.>*/


void utils_loadbar(int x, int n, int r, int w);
/*<Prints progress bar on terminal>*/


float utils_max(const float *v, const int n);
/*<Finding the maximum of a vector v with size n.>*/


float utils_min(const float *v, const int n);
/*<Finding the minimum of a vector v with size n.>*/

#endif 
