#ifndef PFAC_FILE_H_
#define PFAC_FILE_H_

#include "pfac.h"

// #ifdef __cplusplus
// extern "C" {
// #endif   // __cplusplus

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS             if operation is successful
 *  PFAC_STATUS_INVALID_HANDLE      if "handle" is a NULL pointer,
 *                                  please call PFAC_create() to create a legal handle
 *  PFAC_STATUS_INVALID_PARAMETER   if "filename" is a NULL pointer. 
 *                                  The library does not support patterns from standard input
 *  PFAC_STATUS_FILE_OPEN_ERROR     if file "filename" does not exist
 *  PFAC_STATUS_ALLOC_FAILED         
 *  PFAC_STATUS_CUDA_ALLOC_FAILED   if host (device) memory is not enough to parse pattern file.
 *                                  The pattern file is too large to allocate host(device) memory.
 *                                  Please split the pattern file into smaller and try again
 *  PFAC_STATUS_INTERNAL_ERROR      please report bugs
 *  
 */
PFAC_status_t  PFAC_readPatternFromFile( PFAC_handle_t handle, char *filename );

/*
 *  parse pattern file "patternFileName",
 *  (1) store all patterns in "patternPool" and
 *  (2) reorder the patterns according to lexicographic order and store
 *      reordered pointer in "rowPtr"
 *  (3) record original pattern ID in "patternID_table = *patternID_table_ptr"
 *  (4) record pattern length in "patternLen_table = *patternLen_table_ptr"
 *
 *  (5) *pattern_num_ptr = number of patterns
 *  (6) *max_state_num_ptr = estimation (upper bound) of total states in PFAC DFA
 *
 */
PFAC_status_t parsePatternFile( char *patternFileName, char ***rowPtr, char **patternPool,
    int **patternID_table_ptr, int **patternLen_table_ptr, int *max_state_num_ptr, int *pattern_num_ptr ) ;

// #ifdef __cplusplus
// }
// #endif   // __cplusplus


#endif   // PFAC_FILE_H_