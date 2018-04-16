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

PFAC_status_t parsePatternFile( PFAC_handle_t handle, char *patternfilename );

// #ifdef __cplusplus
// }
// #endif   // __cplusplus


#endif   // PFAC_FILE_H_