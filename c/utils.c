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
