// To generate the segment list file

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: ./execute [filename] [number of segments] [seglength]\n");
    exit(1);
  } // if argc

  long segnum, seglength;
  long start, end;

  segnum = atol(argv[2]);
  seglength = atol(argv[3]);

  FILE *F;
  F = fopen(argv[1], "wb");

  for (long i = 0; i < segnum; ++i) {
    start = i * seglength;
    end = (i + 1) * seglength - 1;
    fwrite(&start, sizeof(long), 1, F);
    fwrite(&end, sizeof(long), 1, F);
  } // for

  fclose(F);
} // main()
