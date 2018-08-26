#include <algorithm>
#include <string.h>

#include "pfac.h"
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

    int ret = pfacCompile((PFAC_STRUCT*)handle, NULL, NULL);
    if (ret != 0)
    {
        return PFAC_STATUS_PATTERNS_NOT_READY;
    }
    
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

    char s[12];
    for (int i = 0; i < 10; ++i)
    {
        fscanf(fpin, "%s", s);
        if (pfacAddPattern((PFAC_STRUCT*)handle, (uint8_t*)s, strlen(s), 0, 0, 0, 0, NULL, 0) != 0)
        {
            PFAC_PRINTF("Error: Add pattern \"%s\" failed.", s);
        }
    }

    return PFAC_STATUS_SUCCESS;
}