#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void *alloc1 (size_t n1, size_t size)
	/*< allocate a 1-d array >*/
{
	void *p;

	if ((p=malloc(n1*size))==NULL)
		return NULL;
	return p;
}

void free1 (void *p)
	/*< free a 1-d array >*/
{
	free(p);
}

void **alloc2 (size_t n1, size_t n2, size_t size)
	/*< allocate a 2-d array >*/
{
	size_t i2;
	void **p;

	if ((p=(void**)malloc(n2*sizeof(void*)))==NULL) 
		return NULL;
	if ((p[0]=(void*)malloc(n2*n1*size))==NULL) {
		free(p);
		return NULL;
	}
	for (i2=0; i2<n2; i2++)
		p[i2] = (char*)p[0]+size*n1*i2;
	return p;
}

void free2 (void **p)
	/*< free a 2-d array >*/
{
	free(p[0]);
	free(p);
}

void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size)
	/*< allocate a 3-d array >*/
{
	size_t i3,i2;
	void ***p;

	if ((p=(void***)malloc(n3*sizeof(void**)))==NULL)
		return NULL;
	if ((p[0]=(void**)malloc(n3*n2*sizeof(void*)))==NULL) {
		free(p);
		return NULL;
	}
	if ((p[0][0]=(void*)malloc(n3*n2*n1*size))==NULL) {
		free(p[0]);
		free(p);
		return NULL;
	}
	for (i3=0; i3<n3; i3++) {
		p[i3] = p[0]+n2*i3;
		for (i2=0; i2<n2; i2++)
			p[i3][i2] = (char*)p[0][0]+size*n1*(i2+n2*i3);
	}
	return p;
}

void free3 (void ***p)
	/*< free a 3-d array >*/
{
	free(p[0][0]);
	free(p[0]);
	free(p);
}

//===================1D=====================
int *alloc_int_1d(size_t n1)
	/*< allocate a 1-d array of ints >*/
{
	return (int*)alloc1(n1,sizeof(int));
}

float *alloc_float_1d(size_t n1)
	/*< allocate a 1-d array of floats >*/
{
	return (float*)alloc1(n1,sizeof(float));
}

double *alloc_double_1d(size_t n1)
	/*< allocate a 1-d array of double >*/
{
	return (double*)alloc1(n1,sizeof(double));
}

void free_int_1d(int *p)
	/*< free a 1-d array of ints >*/
{
	free1(p);
}

void free_float_1d(float *p)
	/*< free a 1-d array of floats >*/
{
	free1(p);
}

void free_double_1d(double *p)
	/*< free a 1-d array of floats >*/
{
	free1(p);
}

void zero_int_1d(int *a, int n1)
/*<zero 1D int document>*/
{
	memset(a, 0, n1 * sizeof(int));
}

void zero_float_1d(float *a, int n1)
/*<zero 1D float document>*/
{
	memset(a, 0, n1 * sizeof(float));
}

void zero_double_1d(double *a, int n1)
/*<zero 1D double document>*/
{
	memset(a, 0, n1 * sizeof(double));
}

//===================2D=====================
/* n1: fast dimension; n2: slow dimension */
int **alloc_int_2d(size_t n1, size_t n2)
	/*< allocate a 2-d array of ints >*/
{
	return (int**)alloc2(n1,n2,sizeof(int));
}

float **alloc_float_2d(size_t n1, size_t n2)
	/*< allocate a 2-d array of float >*/
{
	return (float**)alloc2(n1,n2,sizeof(float));
}

double **alloc_double_2d(size_t n1, size_t n2)
	/*< allocate a 2-d array of double >*/
{
	return (double**)alloc2(n1,n2,sizeof(double));
}

void free_int_2d(int **p)
	/*< free a 2-d array of ints >*/
{
	free2((void**)p);
}

void free_float_2d(float **p)
	/*< free a 2-d array of floats >*/
{
	free2((void**)p);
}

void free_double_2d(double **p)
	/*< free a 2-d array of floats >*/
{
	free2((void**)p);
}

void zero_int_2d(int **a, int n1, int n2)
/*<zero 2D int document n1 is the fast axis>*/
{
	memset(a, 0, n1 * n2 * sizeof(int));
}

void zero_float_2d(float **a, int n1, int n2)
/*<zero 2D float document n1 is the fast axis>*/
{
	memset(a, 0, n1 * n2 * sizeof(float));
}

void zero_double_2d(double **a, int n1, int n2)
/*<zero 2D double document n1 is the fast axis>*/
{
	memset(a, 0, n1 * n2 * sizeof(double));
}

//===================3D=====================
int ***alloc_int_3d(size_t n1, size_t n2, size_t n3)
	/*< allocate a 3-d array of ints >*/
{
	return (int***)alloc3(n1,n2,n3,sizeof(int));
}

float ***alloc_float_3d(size_t n1, size_t n2, size_t n3)
	/*< allocate a 3-d array of floats >*/
{
	return (float***)alloc3(n1,n2,n3,sizeof(float));
}

double ***alloc_double_3d(size_t n1, size_t n2, size_t n3)
	/*< allocate a 3-d array of doubles >*/
{
	return (double***)alloc3(n1,n2,n3,sizeof(double));
}

void free_int_3d(int ***p)
	/*< free a 3-d array of ints >*/
{
	free3((void***)p);
}

void free_float_3d(float ***p)
	/*< free a 3-d array of floats >*/
{
	free3((void***)p);
}

void free_double_3d(double ***p)
	/*< free a 3-d array of doubles >*/
{
	free3((void***)p);
}

void zero_int_3d(int ***a, int n1, int n2, int n3)
/*<zero 2D int document n1 is the fast axis>*/
{
	memset(a, 0, n1 * n2 * n3 * sizeof(int));
}

void zero_float_3d(float ***a, int n1, int n2, int n3)
/*<zero 2D float document n1 is the fast axis>*/
{
	memset(a, 0, n1 * n2 * n3 * sizeof(float));
}

void zero_double_3d(double ***a, int n1, int n2, int n3)
/*<zero 2D double document n1 is the fast axis>*/
{
	memset(a, 0, n1 * n2 * n3 * sizeof(double));
}
