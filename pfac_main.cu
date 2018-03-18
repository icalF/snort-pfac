#include "pfac.h"
#include "pfac_match.h"
#include "pfac_file.h"
#include "pfac_table.h"

#include <cassert>

int main(int argc, char **argv)
{
    char dumpTableFile[] = "table.txt";
    char inputFile[2][40] = {"test/data/example_input", "test/data/example_input2"};
    char patternFile[] = "test/pattern/example_pattern";
    PFAC_handle_t handles[2];
    PFAC_status_t PFAC_status;
    int input_size[2];
    char *h_inputString[2] = {NULL,NULL};
    int  *h_matched_result[2] = {NULL,NULL};

    // step 1: create PFAC handle 
    for (int i = 0; i < 2; ++i) {
        PFAC_status = PFAC_create( &handles[i] ) ;
        if ( PFAC_STATUS_SUCCESS != PFAC_status ){
            printf("Error: fails to create handle %d, %s\n", i, PFAC_getErrorString(PFAC_status) );
            exit(1) ;   
        }
    }

    // step 2: read patterns and dump transition table 
    for (int i = 0; i < 2; ++i) {
        PFAC_status = PFAC_readPatternFromFile( handles[i], patternFile) ;
        if ( PFAC_STATUS_SUCCESS != PFAC_status ){
            printf("Error: fails to read pattern %d from file, %s\n", i, PFAC_getErrorString(PFAC_status) );
            exit(1) ;   
        }
    }

    // dump transition table 
     FILE *table_fp = fopen(dumpTableFile, "w");
     assert(NULL != table_fp);
     PFAC_status = PFAC_dumpTransitionTable( handles[0], table_fp );
     fclose(table_fp);
     if (PFAC_STATUS_SUCCESS != PFAC_status) {
         printf("Error: fails to dump transition table, %s\n", PFAC_getErrorString(PFAC_status));
         exit(1);
     }

    //step 3: prepare input stream
    for (int i = 0; i < 2; ++i) {
        FILE* fpin = fopen(inputFile[i], "rb");
        assert(NULL != fpin);

        // obtain file size
        fseek(fpin, 0, SEEK_END);
        input_size[i] = ftell(fpin);
        rewind(fpin);

        // allocate memory to contain the whole file
        h_inputString[i] = (char *)malloc(sizeof(char)*input_size[i]);
        assert(NULL != h_inputString[i]);

        h_matched_result[i] = (int *)malloc(sizeof(int)*input_size[i]);
        assert(NULL != h_matched_result[i]);
        memset(h_matched_result[i], 0, sizeof(int)*input_size[i]);

        // copy the file into the buffer
        input_size[i] = fread(h_inputString[i], 1, input_size[i], fpin);
        fclose(fpin);
    }

    // step 4: run PFAC on GPU           
    for (int i = 0; i < 2; ++i) {
        PFAC_status = PFAC_matchFromHost( handles[i], h_inputString[i], input_size[i], h_matched_result[i] ) ;
        if ( PFAC_STATUS_SUCCESS != PFAC_status ){
            printf("Error: fails to PFAC_matchFromHost, %s\n", PFAC_getErrorString(PFAC_status) );
            exit(1) ;   
        }     
    }

    // step 5: output matched result
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < input_size[i]; j++) {
            if (h_matched_result[i][j] != 0) {
                printf("At position %4d, match pattern %d\n", j, h_matched_result[i][j]);
            }
        }
        puts("");
    }

    for (int i = 0; i < 2; ++i) {
        PFAC_status = PFAC_destroy( handles[i] ) ;
        assert( PFAC_STATUS_SUCCESS == PFAC_status );
    
        free(h_inputString[i]);
        free(h_matched_result[i]);
    }

    return 0;
}