#include "pfac.h"
#include "pfac_file.h"

#include <cassert>
#include <cstdio>
#include <cstring>

int MatchFound (void * id, void *tree, int index, void *data, void *neg_list)
{
    fprintf (stdout, "%s\n", (char *) id);
    return 0;
}

int main(int argc, char **argv)
{
    char dumpTableFile[] = "table.txt";
    char inputFile[] = "/home/xtremax/spfac-nv/test/data/example_input";
    char patternFile[] = "/home/xtremax/spfac-nv/test/pattern/example_pattern";
    PFAC_STRUCT *pfac;
    PFAC_status_t PFAC_status;
    int input_size;
    char *h_inputString = NULL;
    int  *h_matched_result = NULL;

    // step 1: create PFAC handle 
    pfac = pfacNew( NULL, NULL, NULL ) ;
    assert( pfac != NULL );

    // step 2: read patterns and dump transition table 
    PFAC_status = PFAC_readPatternFromFile( pfac, patternFile) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to read pattern from file, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;   
    }

    // dump transition table 
    // FILE *table_fp = fopen(dumpTableFile, "w");
    // assert(NULL != table_fp);
    // PFAC_status = PFAC_dumpTransitionTable( pfac, table_fp );
    // fclose(table_fp);
    // if ( PFAC_STATUS_SUCCESS != PFAC_status ) {
    //     printf("Error: fails to dump transition table, %s\n", PFAC_getErrorString(PFAC_status));
    //     exit(1);
    // }

    //step 3: prepare input stream
    FILE* fpin = fopen(inputFile, "rb");
    assert(NULL != fpin);

    // obtain file size
    fseek(fpin, 0, SEEK_END);
    input_size = ftell(fpin);
    rewind(fpin);

    // allocate memory to contain the whole file
    h_inputString = (char *)malloc(sizeof(char)*input_size);
    assert(NULL != h_inputString);

    h_matched_result = (int *)malloc(sizeof(int)*input_size);
    assert(NULL != h_matched_result);
    memset(h_matched_result, 0, sizeof(int)*input_size);

    // copy the file into the buffer
    input_size = fread(h_inputString, 1, input_size, fpin);
    fclose(fpin);

    // step 4: run PFAC on GPU
    int current_state = 0;           
    int count = pfacSearch ( pfac, (unsigned char*) h_inputString, input_size, MatchFound, NULL, &current_state );

    // step 5: output matched result
    // for (int i = 0; i < input_size; i++) {
    //     if (h_matched_result[i] != 0) {
    //         printf("At position %4d, match pattern %d\n", i, h_matched_result[i]);
    //     }
    // }
    printf("Pattern found: %d\n", count);

    pfacFree( pfac ) ;

    /*
     * Address consistency
     * https://stackoverflow.com/questions/6054271/freeing-pointers-from-inside-other-functions-in-c
     */
    pfac = NULL;

    return 0;
}
