/******************************************************************
  ALLOC - Allocate and free multi-dimensional arrays

  alloc1		allocate a 1-d array
  realloc1	re-allocate a 1-d array
  free1		free a 1-d array
  alloc2		allocate a 2-d array
  free2		free a 2-d array
  alloc3		allocate a 3-d array
  free3		free a 3-d array
  alloc4		allocate a 4-d array
  free4		free a 4-d array
  alloc5		allocate a 5-d array
  free5		free a 5-d array
  alloc6		allocate a 6-d array
  free6		free a 6-d arrayalloc1int	
  allocate a 1-d array of ints
  realloc1int	re-allocate a 1-d array of ints
  free1int	free a 1-d array of ints
  alloc2int	allocate a 2-d array of ints
  free2int	free a 2-d array of ints
  alloc3int	allocate a 3-d array of ints
  free3int	free a 3-d array of ints
  alloc1float	allocate a 1-d array of floats
  realloc1float	re-allocate a 1-d array of floats
  free1float	free a 1-d array of floats
  alloc2float	allocate a 2-d array of floats
  free2float	free a 2-d array of floats
  alloc3float	allocate a 3-d array of floats
  free3float	free a 3-d array of floats
  alloc4float	allocate a 4-d array of floats 
  free4float      free a 4-d array of floats 
  alloc5float     allocate a 5-d array of floats 
  free5float      free a 5-d array of floats 
  alloc6float     allocate a 6-d array of floats 
  free6float      free a 6-d array of floats 
  alloc4int       allocate a 4-d array of ints 
  free4int        free a 4-d array of ints 
  alloc5int       allocate a 5-d array of ints 
  free5int        free a 5-d array of ints 
  alloc5uchar	allocate a 5-d array of unsigned chars 
  free5uchar	free a 5-d array of unsiged chars 
  alloc2ushort    allocate a 2-d array of unsigned shorts 
  free2ushort     free a 2-d array of unsiged shorts
  alloc3ushort    allocate a 3-d array of unsigned shorts 
  free3ushort     free a 3-d array of unsiged shorts
  alloc5ushort    allocate a 5-d array of unsigned shorts 
  free5ushort     free a 5-d array of unsiged shorts
  alloc6ushort    allocate a 6-d array of unsigned shorts 
  free6ushort     free a 6-d array of unsiged shorts
  alloc1double	allocate a 1-d array of doubles
  realloc1double	re-allocate a 1-d array of doubles
  free1double	free a 1-d array of doubles
  alloc2double	allocate a 2-d array of doubles
  free2double	free a 2-d array of doubles
  alloc3double	allocate a 3-d array of doubles
  free3double	free a 3-d array of doubles
  alloc1complex	allocate a 1-d array of complexs
  realloc1complex	re-allocate a 1-d array of complexs
  free1complex	free a 1-d array of complexs
  alloc2complex	allocate a 2-d array of complexs
  free2complex	free a 2-d array of complexs
  alloc3complex	allocate a 3-d array of complexs
  free3complex	free a 3-d array of complexs
  alloc4complex	allocate a 4-d array of complexs
  free4complex	free a 4-d array of complexs
  alloc5complex	allocate a 5-d array of complexs
  free5complex	free a 5-d array of complexs

  zero1int        initialize the 1-d int array with zero
  zero2int        initialize the 2-d int array with zero
zero3int        initialize the 3-d int array with zero

zero1float      initialize the 1-d float array with zero
zero2float      initialize the 2-d float array with zero
zero3float      initialize the 3-d float array with zero
zero4float      initialize the 4-d float array with zero

zero1double     initialize the 1-d double array with zero
zero2double     initialize the 2-d double array with zero
zero3double     initialize the 3-d double array with zero

zero1complex    initialize the 1-d complex array with zero
zero2complex    initialize the 2-d complex array with zero
zero3complex    initialize the 3-d complex array with zero
zero4complex    initialize the 4-d complex array with zero
zero5complex    initialize the 5-d complex array with zero

******************************************************************************
Notes:
The functions defined below are intended to simplify manipulation
of multi-dimensional arrays in scientific programming in C.  These
functions are useful only because true multi-dimensional arrays
in C cannot have variable dimensions (as in FORTRAN).  For example,
   the following function IS NOT valid in C:
void badFunc(a,n1,n2)
	float a[n2][n1];
{
	a[n2-1][n1-1] = 1.0;
}
However, the following function IS valid in C:
void goodFunc(a,n1,n2)
	float **a;
{
	a[n2-1][n1-1] = 1.0;
}
Therefore, the functions defined below do not allocate true
multi-dimensional arrays, as described in the C specification.
Instead, they allocate and initialize pointers (and pointers to 
		pointers) so that, for example, a[i2][i1] behaves like a 2-D array.

The array dimensions are numbered, which makes it easy to add 
functions for arrays of higher dimensions.  In particular,
		  the 1st dimension of length n1 is always the fastest dimension,
		  the 2nd dimension of length n2 is the next fastest dimension,
		  and so on.  Note that the 1st (fastest) dimension n1 is the 
		  first argument to the allocation functions defined below, but 
	that the 1st dimension is the last subscript in a[i2][i1].
(This is another important difference between C and Fortran.)

	The allocation of pointers to pointers implies that more storage
	is required than is necessary to hold a true multi-dimensional array.
	The fraction of the total storage allocated that is used to hold 
	pointers is approximately 1/(n1+1).  This extra storage is unlikely
	to represent a significant waste for large n1.

	The functions defined below are significantly different from similar 
	functions described by Press et al, 1988, Numerical Recipes in C.
	In particular, the functions defined below:
	(1) Allocate arrays of arbitrary size elements.
	(2) Allocate contiguous storage for arrays.
	(3) Return NULL if allocation fails (just like malloc).
	(4) Do not provide arbitrary lower and upper bounds for arrays.

	Contiguous storage enables an allocated multi-dimensional array to
	be passed to a C function that expects a one-dimensional array.
	For example, to allocate and zero an n1 by n2 two-dimensional array
	of floats, one could use
	a = alloc2(n1,n2,sizeof(float));
	zeroFloatArray(n1*n2,a[0]);
	where zeroFloatArray is a function defined as
void zeroFloatArray(int n, float *a)
{
	int i;
	for (i=0; i<n; i++)
		a[i] = 0.0;
}

Internal error handling and arbitrary array bounds, if desired,
		 should be implemented in functions that call the functions defined 
		 below, with the understanding that these enhancements may limit 
		 portability.
		 **************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

void *alloc1 (size_t n1, size_t size)
	/*< allocate a 1-d array >*/
{
	void *p;

	if ((p=malloc(n1*size))==NULL)
		return NULL;
	return p;
}

void *realloc1(void *v, size_t n1, size_t size)
	/*< re-allocate a 1-d array >*/
{
	void *p;

	if ((p=realloc(v,n1*size))==NULL)
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

void ****alloc4 (size_t n1, size_t n2, size_t n3, size_t n4, size_t size)
	/*< allocate a 4-d array >*/
{
	size_t i4,i3,i2;
	void ****p;

	if ((p=(void****)malloc(n4*sizeof(void***)))==NULL)
		return NULL;
	if ((p[0]=(void***)malloc(n4*n3*sizeof(void**)))==NULL) {
		free(p);
		return NULL;
	}
	if ((p[0][0]=(void**)malloc(n4*n3*n2*sizeof(void*)))==NULL) {
		free(p[0]);
		free(p);
		return NULL;
	}
	if ((p[0][0][0]=(void*)malloc(n4*n3*n2*n1*size))==NULL) {
		free(p[0][0]);
		free(p[0]);
		free(p);
		return NULL;
	}
	for (i4=0; i4<n4; i4++) {
		p[i4] = p[0]+i4*n3;
		for (i3=0; i3<n3; i3++) {
			p[i4][i3] = p[0][0]+n2*(i3+n3*i4);
			for (i2=0; i2<n2; i2++)
				p[i4][i3][i2] = (char*)p[0][0][0]+
					size*n1*(i2+n2*(i3+n3*i4));
		}
	}
	return p;
}

void free4 (void ****p)
	/*< free a 4-d array >*/
{
	free(p[0][0][0]);
	free(p[0][0]);
	free(p[0]);
	free(p);
}

int *alloc1int(size_t n1)
	/*< allocate a 1-d array of ints >*/
{
	return (int*)alloc1(n1,sizeof(int));
}

int *realloc1int(int *v, size_t n1)
	/*< re-allocate a 1-d array of ints >*/
{
	return (int*)realloc1(v,n1,sizeof(int));
}

void free1int(int *p)
	/*< free a 1-d array of ints >*/
{
	free1(p);
}

/* n1: fast dimension; n2: slow dimension */
int **alloc2int(size_t n1, size_t n2)
	/*< allocate a 2-d array of ints >*/
{
	return (int**)alloc2(n1,n2,sizeof(int));
}

void free2int(int **p)
	/*< free a 2-d array of ints >*/
{
	free2((void**)p);
}

int ***alloc3int(size_t n1, size_t n2, size_t n3)
	/*< allocate a 3-d array of ints >*/
{
	return (int***)alloc3(n1,n2,n3,sizeof(int));
}

void free3int(int ***p)
	/*< free a 3-d array of ints >*/
{
	free3((void***)p);
}

float *alloc1float(size_t n1)
	/*< allocate a 1-d array of floats >*/
{
	return (float*)alloc1(n1,sizeof(float));
}

float *realloc1float(float *v, size_t n1)
	/*< re-allocate a 1-d array of floats >*/
{
	return (float*)realloc1(v,n1,sizeof(float));
}

void free1float(float *p)
	/*< free a 1-d array of floats >*/
{
	free1(p);
}

/* allocate a 2-d array of floats */
float **alloc2float(size_t n1, size_t n2)
	/*<  n1: fast dimension; n2: slow dimension >*/
{
	return (float**)alloc2(n1,n2,sizeof(float));
}

void free2float(float **p)
	/*< free a 2-d array of floats >*/
{
	free2((void**)p);
}

float ***alloc3float(size_t n1, size_t n2, size_t n3)
	/*< allocate a 3-d array of floats >*/
{
	return (float***)alloc3(n1,n2,n3,sizeof(float));
}

void free3float(float ***p)
	/*< free a 3-d array of floats >*/
{
	free3((void***)p);
}

float ****alloc4float(size_t n1, size_t n2, size_t n3, size_t n4)
	/*< allocate a 4-d array of floats, added by Zhaobo Meng, 1997 >*/
{
	return (float****)alloc4(n1,n2,n3,n4,sizeof(float));
}

void free4float(float ****p)
	/*< free a 4-d array of floats, added by Zhaobo Meng, 1997 >*/
{
	free4((void****)p);
}

int ****alloc4int(size_t n1, size_t n2, size_t n3, size_t n4)
/*< allocate a 4-d array of ints, added by Zhaobo Meng, 1997 >*/
{
	return (int****)alloc4(n1,n2,n3,n4,sizeof(int));
}

void free4int(int ****p)
/*< free a 4-d array of ints, added by Zhaobo Meng, 1997 >*/
{
	free4((void****)p);
}

unsigned char ****alloc4uchar(size_t n1, size_t n2, size_t n3, size_t n4)
/*< allocate a 4-d array of chars, added by Cheng Jiubing >*/
{
	return (unsigned char****)alloc4(n1,n2,n3,n4,sizeof(unsigned char));
}

void free4uchar(unsigned char ****p)
/*< free a 4-d array of chars, added by Cheng Jiubing >*/
{
	free4((void****)p);
}

unsigned char ***alloc3uchar(size_t n1, size_t n2, size_t n3)
/*< allocate a 3-d array of chars, added by Cheng Jiubing >*/
{
	return (unsigned char***)alloc3(n1,n2,n3,sizeof(unsigned char));
}

void free3uchar(unsigned char ***p)
/*< free a 3-d array of chars, added by Cheng Jiubing >*/
{
	free3((void***)p);
}

unsigned char **alloc2uchar(size_t n1, size_t n2)
/*< allocate a 2-d array of chars, added by Cheng Jiubing >*/
{
	return (unsigned char**)alloc2(n1,n2,sizeof(unsigned char));
}

void free2uchar(unsigned char **p)
/*< free a 2-d array of chars, added by Cheng Jiubing >*/
{
	free2((void**)p);
}

char **alloc2char(size_t n1, size_t n2)
/*< allocate a 2-d array of chars, added by Cheng Jiubing >*/
{
	return (char**)alloc2(n1,n2,sizeof(char));
}

void free2char(char **p)
/*< free a 2-d array of chars, added by Cheng Jiubing >*/
{
	free2((void**)p);
}

unsigned char *alloc1uchar(size_t n1)
/*< allocate a 1-d array of chars, added by Cheng Jiubing >*/
{
	return (unsigned char*)alloc1(n1,sizeof(unsigned char));
}

void free1uchar(unsigned char *p)
/*< free a 1-d array of chars, added by Cheng Jiubing >*/
{
	free1((void*)p);
}

unsigned short ***alloc3ushort(size_t n1, size_t n2, size_t n3)
/*< allocate a 3-d array of ints, added by Meng, 1997 >*/
{
	return (unsigned short***)alloc3(n1,n2,n3,sizeof(unsigned short));
}

unsigned short **alloc2ushort(size_t n1, size_t n2)
/*< allocate a 2-d array of ints, added by Meng, 1997 >*/
{
	return (unsigned short**)alloc2(n1,n2,sizeof(unsigned short));
}

void free3ushort(unsigned short ***p)
/*< free a 3-d array of shorts, added by Zhaobo Meng, 1997 >*/
{
	free3((void***)p);
}

void free2ushort(unsigned short **p)
/*< free a 2-d array of shorts, added by Zhaobo Meng, 1997 >*/
{
	free2((void**)p);
}

double *alloc1double(size_t n1)
/*< allocate a 1-d array of doubles >*/
{
	return (double*)alloc1(n1,sizeof(double));
}

double *realloc1double(double *v, size_t n1)
/*< re-allocate a 1-d array of doubles >*/
{
	return (double*)realloc1(v,n1,sizeof(double));
}

void free1double(double *p)
/*< free a 1-d array of doubles >*/
{
	free1(p);
}

double **alloc2double(size_t n1, size_t n2)
/*< allocate a 2-d array of doubles >*/
{
	return (double**)alloc2(n1,n2,sizeof(double));
}

void free2double(double **p)
/*< free a 2-d array of doubles >*/
{
	free2((void**)p);
}

double ***alloc3double(size_t n1, size_t n2, size_t n3)
/*< allocate a 3-d array of doubles >*/
{
	return (double***)alloc3(n1,n2,n3,sizeof(double));
}

void free3double(double ***p)
/*< free a 3-d array of doubles >*/
{
	free3((void***)p);
}

/**************************************************************/
#ifdef TEST1
main()
{
	short   *hv, **hm;
	int     *iv, **im;
	float   *fv, **fm;
	double  *dv, **dm;
	size_t i1, i2;
	size_t n1, n2;

	scanf("%d %*[^\n]", &n1);
	scanf("%d %*[^\n]", &n2);

	/* Exercise 1-d routines */
	hv = (short *) alloc1(n1, sizeof(short));
	iv = alloc1int(n1);
	fv = alloc1float(n1);
	dv = alloc1double(n1);

	for (i1 = 0; i1 < n1; ++i1) {
		hv[i1] = i1;
		iv[i1] = i1;
		fv[i1]  = (float) i1;
		dv[i1] = (double) i1;
	}

	printf("short vector:\n");
	for (i1 = 0; i1 < n1; ++i1) {
		printf("hv[%d] = %hd\n", i1, hv[i1]);
	}
	putchar('\n');

	printf("int vector:\n");
	for (i1 = 0; i1 < n1; ++i1) {
		printf("iv[%d] = %d\n", i1, iv[i1]);
	}
	putchar('\n');

	printf("float vector:\n");
	for (i1 = 0; i1 < n1; ++i1) {
		printf("fv[%d] = %f\n", i1, fv[i1]);
	}
	putchar('\n');

	printf("double vector:\n");
	for (i1 = 0; i1 < n1; ++i1) {
		printf("dv[%d] = %lf\n", i1, dv[i1]);
	}
	putchar('\n');


	free1(hv);
	free1int(iv);
	free1float(fv);
	free1double(dv);


	/* Exercise 2-d routines */
	hm = (short *) alloc2(n1, n2, sizeof(short));
	im = alloc2int(n1, n2);
	fm = alloc2float(n1, n2);
	dm = alloc2double(n1, n2);

	for (i2 = 0; i2 < n2; ++i2) {
		for (i1 = 0; i1 < n1; ++i1) {
			hm[i2][i1] = i1 + 2*i2;
			im[i2][i1] = i1 + 2*i2;
			fm[i2][i1] = (float) (i1 + 2*i2);
			dm[i2][i1] = (double) (i1 + 2*i2);
		}
	}

	printf("short matrix:\n");
	for (i2 = 0; i2 < n2; ++i2) {
		for (i1 = 0; i1 < n1; ++i1) {
			printf("hm[%d, %d] = %hd ", i2, i1, hm[i2][i1]);
		}
		putchar('\n');
	}
	putchar('\n');

	printf("int matrix:\n");
	for (i2 = 0; i2 < n2; ++i2) {
		for (i1 = 0; i1 < n1; ++i1) {
			printf("im[%d, %d] = %d ", i2, i1, im[i2][i1]);
		}
		putchar('\n');
	}
	putchar('\n');

	printf("float matrix:\n");
	for (i2 = 0; i2 < n2; ++i2) {
		for (i1 = 0; i1 < n1; ++i1) {
			printf("fm[%d, %d] = %f ", i2, i1, fm[i2][i1]);
		}
		putchar('\n');
	}
	putchar('\n');

	printf("double matrix:\n");
	for (i2 = 0; i2 < n2; ++i2) {
		for (i1 = 0; i1 < n1; ++i1) {
			printf("dm[%d, %d] = %lf ", i2, i1, dm[i2][i1]);
		}
		putchar('\n');
	}
	putchar('\n');

	free2(hm);
	free2int(im);
	free2float(fm);
	free2double(dm);

	exit(0);
}
#endif
