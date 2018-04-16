#include <algorithm>

#include "pfac_file.h"
#include "pfac_table.h"

/*
 *  if return status is not PFAC_STATUS_SUCCESS, then all reousrces are free.
 */
PFAC_status_t  PFAC_readPatternFromFile( PFAC_handle_t handle, char *filename )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( NULL == filename ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( handle->isPatternsReady ) {
        // free previous patterns, including transition tables in host and device memory
        PFAC_freeResource( handle );
    }

    PFAC_status_t PFAC_status = parsePatternFile( handle, filename ) ;

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        PFAC_freeResource( handle );
        return PFAC_status ;
    }
    
    return PFAC_prepareTable(handle);
}

PFAC_status_t parsePatternFile( PFAC_handle_t handle, char *patternfilename )
{
    if (NULL == patternfilename) {
        return PFAC_STATUS_INVALID_PARAMETER;
    }

    FILE* fpin = fopen(patternfilename, "rb");
    if (fpin == NULL) {
        PFAC_PRINTF("Error: Open pattern file %s failed.", patternfilename);
        return PFAC_STATUS_FILE_OPEN_ERROR;
    }

    // step 1: find size of the file
    // obtain file size
    fseek(fpin, 0, SEEK_END);
    int file_size = ftell(fpin);
    handle->max_numOfStates = file_size;
    rewind(fpin);

    // step 2: allocate a buffer to contains all patterns
    handle->valPtr = (char*)malloc(sizeof(char)*file_size);
    if (NULL == handle->valPtr) {
        return PFAC_STATUS_ALLOC_FAILED;
    }

    // copy the file into the buffer
    file_size = fread(handle->valPtr, 1, file_size, fpin);
    fclose(fpin);

    handle->numOfPatterns = 10;

    PFAC_status_t status = PFAC_fillPatternTable(handle);
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_fillPatternTable, %s\n", PFAC_getErrorString(status) );
        return status;
    }

    return PFAC_STATUS_SUCCESS;
}