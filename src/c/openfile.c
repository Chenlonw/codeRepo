#include<stdlib.h>
#include<stdio.h>

FILE *sfopen(const char *fn, const char *stat)
/*<safely open the files>*/
{
    FILE *fp;
    if ((fp=fopen(fn, stat))==NULL)
    {
        fprintf(stderr, "=======Caution=======\n");
        fprintf(stderr, "cannot open file %s\n", fn);
        exit(0);
    }
    else
        return fp;
}
