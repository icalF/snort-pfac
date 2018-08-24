#include <cuda.h>

#include "pfac_match.h"
#include "pfac_table.h"
#include <cuda_runtime_api.h>

/*
 *  platform is immaterial, do matching on GPU
 *
 *  WARNING: d_input_string is allocated by caller, the size may not be multiple of 4.
 *  if shared mmeory version is chosen (for example, maximum pattern length is less than 512), then
 *  it is out-of-array bound logically, but it may not happen physically because basic unit of cudaMalloc() 
 *  is 256 bytes.  
 */
PFAC_status_t  PFAC_matchFromDevice( PFAC_handle_t handle, size_t input_size )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }
    if ( NULL == handle->d_matched_result ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == handle->d_num_matched ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( 0 == input_size ){ 
        return PFAC_STATUS_SUCCESS ;    
    }

    correctTextureMode(handle);
    
    PFAC_status_t PFAC_status ;
    PFAC_status = (*(handle->kernel_ptr))( handle, handle->d_input_string, input_size, handle->d_matched_result, handle->d_num_matched );
    return PFAC_status;
}

PFAC_status_t  PFAC_matchFromHost( PFAC_handle_t handle, size_t input_size )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }
    if ( NULL == handle->h_input_string ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == handle->d_input_string ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == handle->h_matched_result ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( 0 == input_size ){ 
        return PFAC_STATUS_SUCCESS ;    
    }

    // copy input string from host to device
    cudaError_t cuda_status1 = cudaMemcpy(handle->d_input_string, handle->h_input_string, input_size, cudaMemcpyHostToDevice);
    cudaError_t cuda_status2 = cudaMemset(handle->d_num_matched, 0, THREAD_BLOCK_SIZE*sizeof(int));
    if ( cudaSuccess != cuda_status1 || cudaSuccess != cuda_status2 ) {
        cudaFree(handle->d_input_string); 
        cudaFree(handle->d_matched_result);
        cudaFree(handle->d_num_matched);
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    PFAC_status_t PFAC_status = PFAC_matchFromDevice(handle, input_size);

    if ( PFAC_STATUS_SUCCESS != PFAC_status ) {
        return PFAC_status ;
    }

    // copy the result data from device to host
    cuda_status1 = cudaMemcpy(handle->h_matched_result, handle->d_matched_result, input_size*sizeof(int), cudaMemcpyDeviceToHost);
    cuda_status2 = cudaMemcpy(handle->h_num_matched, handle->d_num_matched, THREAD_BLOCK_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    if ( cudaSuccess != cuda_status1 || cudaSuccess != cuda_status2 ) {
        return PFAC_STATUS_INTERNAL_ERROR;
    }

    return PFAC_STATUS_SUCCESS ;
}
