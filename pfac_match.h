#ifndef PFAC_MATCH_H_
#define PFAC_MATCH_H_

#include "pfac.h"

// #ifdef __cplusplus
// extern "C" {
// #endif   // __cplusplus

/*
 *  platform is immaterial, do matching on GPU
 *
 *  WARNING: d_input_string is allocated by caller, the size may not be multiple of 4.
 *  if shared mmeory version is chosen (for example, maximum pattern length is less than 512), then
 *  it is out-of-array bound logically, but it may not happen physically because basic unit of cudaMalloc() 
 *  is 256 bytes.  
 */
PFAC_status_t  PFAC_matchFromDevice( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result );
PFAC_status_t  PFAC_matchFromHost( PFAC_handle_t handle, char *h_input_string, size_t input_size,
    int *h_matched_result );

/* KERNEL */

extern PFAC_status_t  PFAC_kernel_timeDriven_wrapper( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result) ;


// #ifdef __cplusplus
// }
// #endif   // __cplusplus


#endif   // PFAC_MATCH_H_