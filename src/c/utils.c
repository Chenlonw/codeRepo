#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

void utils_print_title(const char *title)
/*<Printing out program title to terminal.>*/
{
	int i;
	int len;
	len=strlen(title);
	len=len+16;
	fprintf(stderr,"\n");
	for (i=0; i<len+2; i++) fprintf(stderr, "*");
	fprintf(stderr,"\n");
	fprintf(stderr,"***");
	for (i=0; i<len-4; i++) fprintf(stderr, " ");
	fprintf(stderr,"***\n");
	fprintf(stderr,"***      %s      ***\n",title);
	fprintf(stderr,"***");
	for (i=0; i<len-4; i++) fprintf(stderr, " ");
	fprintf(stderr,"***\n");
	for (i=0; i<len+2; i++) fprintf(stderr, "*");
	fprintf(stderr,"\n");
}

void utils_loadbar(int x, int n, int r, int w)
/*<Prints progress bar on terminal>*/
{
	if ( r > n) r = n;
	// Only update r times.
	if ( x % (n/r) != 0 ) return;

	// Calculuate the ratio of complete-to-incomplete.
	float ratio = x/(float)n;
	int   c     = ratio * w;

	// Show the percentage complete.
	fprintf(stderr,"%3d%% [", (int)(ratio*100) );

	// Show the load bar.
	for (x=0; x<c; x++)
		fprintf(stderr,"=");

	for (x=c; x<w; x++)
		fprintf(stderr," ");

	fprintf(stderr,"]\r"); // Move to the first column
	fflush(stderr);
}

float utils_max(const float *v, const int n)
/*<Finding the maximum of a vector v with size n.>*/
{
	float val = v[0];
	int i;

	for(i=0; i<n; i++) {
		if(v[i]>val) val=v[i];
	}
	return val;
}

float utils_min(const float *v, const int n)
/*<Finding the minimum of a vector v with size n.>*/
{
	float val = v[0];
	int i;

	for(i=0; i<n; i++) {
		if(v[i]<val) val=v[i];
	}
	return val;
}
