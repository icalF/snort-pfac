#include <algorithm>
#include <cassert>
#include <string.h>

#include <sys/stat.h>

#include "pfac.h"
#include "pfac_file.h"
#include "pfac_table.h"

int obtain_file_size(FILE* fpin) 
{
    struct stat stbuf;
    int fd = fileno(fpin);
    if ((fstat(fd, &stbuf) != 0) || (!S_ISREG(stbuf.st_mode))) {
        PFAC_PRINTF("Error: Get file size failed.");
        return 0;
    }

    return stbuf.st_size;
}

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

    // int ret = pfacCompile((PFAC_STRUCT*)handle, NULL, NULL);
    // if (ret != 0)
    // {
    //     return PFAC_STATUS_PATTERNS_NOT_READY;
    // }
    
    return PFAC_STATUS_SUCCESS;
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

    int file_size = obtain_file_size(fpin);
    handle->max_numOfStates = file_size;

    // allocate a buffer to contains all patterns
    handle->valPtr = (char*)malloc(sizeof(char) * file_size + 1);
    if (NULL == handle->valPtr) {
        return PFAC_STATUS_ALLOC_FAILED;
    }

    // copy the file into the buffer
    file_size = fread(handle->valPtr, 1, file_size, fpin);
    fclose(fpin);

    PFAC_status_t status = PFAC_elaboratePatterns(handle);
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_elaboratePatterns, %s\n", PFAC_getErrorString(status) );
        return status;
    }

    status = PFAC_fillPatternTable(handle);
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_fillPatternTable, %s\n", PFAC_getErrorString(status) );
        return status;
    }

    status = PFAC_prepareTable(handle);
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_preparePatternTable, %s\n", PFAC_getErrorString(status) );
        return status;
    }

    return PFAC_STATUS_SUCCESS;
}

PFAC_status_t matchFromFile( PFAC_handle_t handle, const char *filename )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( NULL == filename ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( !handle->isPatternsReady ) {
        return PFAC_STATUS_PATTERNS_NOT_READY;
    }

    // prepare input stream
    FILE* fpin = fopen(filename, "rb");
    if (NULL == fpin)
    {
        return PFAC_STATUS_FILE_OPEN_ERROR;
    }

    // obtain file size
    int file_size = obtain_file_size(fpin);
    fread(handle->h_input_string, 1, file_size, fpin);
    printf("%s Mantab anjeng\n", filename);


    fclose(fpin);
    
    return PFAC_STATUS_SUCCESS;
}

PFAC_status_t PFAC_elaboratePatterns(PFAC_handle_t handle)
{
    char *pat = handle->valPtr;
    char *it = handle->valPtr;
    int count = 0, len = 0;

    while (count < handle->max_numOfStates) {
        count++;
        len++;
        it++;

        if (*it == '\n') {
            pfacAddPattern ( (PFAC_STRUCT*) handle, (uint8_t*) pat, len, 0,
                            0, 0, 0, NULL, 0 );
            
            pat = it + 1;
            len = 0;
        }
    }

    return PFAC_STATUS_SUCCESS;
}