#include <cuda.h>

#include "pfac_match.h"
#include "pfac_table.h"

/*
 *  platform is immaterial, do matching on GPU
 *
 *  WARNING: d_input_string is allocated by caller, the size may not be multiple of 4.
 *  if shared mmeory version is chosen (for example, maximum pattern length is less than 512), then
 *  it is out-of-array bound logically, but it may not happen physically because basic unit of cudaMalloc() 
 *  is 256 bytes.  
 */
PFAC_status_t  PFAC_matchFromDevice( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }
    if ( NULL == d_input_string ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == d_matched_result ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( 0 == input_size ){ 
        return PFAC_STATUS_SUCCESS ;    
    }

    correctTextureMode(handle);
    
    PFAC_status_t PFAC_status ;
    PFAC_status = (*(handle->kernel_ptr))( handle, d_input_string, input_size, d_matched_result );
    return PFAC_status;
}

PFAC_status_t  PFAC_matchFromHost( PFAC_handle_t handle, char *h_input_string, size_t input_size,
    int *h_matched_result )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }
    if ( NULL == h_input_string ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == h_matched_result ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( 0 == input_size ){ 
        return PFAC_STATUS_SUCCESS ;    
    }
    
    char *d_input_string  = NULL;
    int *d_matched_result = NULL;
    int *d_num_matched    = NULL;

    // n_hat = number of integers of input string
    int n_hat = (input_size + sizeof(int)-1)/sizeof(int) ;

    // allocate memory for input string and result
    // basic unit of d_input_string is integer
    cudaError_t cuda_status1 = cudaMalloc((void **) &d_input_string,            n_hat*sizeof(int) );
    cudaError_t cuda_status2 = cudaMalloc((void **) &d_matched_result,     input_size*sizeof(int) );
    cudaError_t cuda_status3 = cudaMalloc((void **) &d_num_matched, THREAD_BLOCK_SIZE*sizeof(int) );
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) || (cudaSuccess != cuda_status3) ){
          if ( NULL != d_input_string   ) { cudaFree(d_input_string); }
          if ( NULL != d_matched_result ) { cudaFree(d_matched_result); }
          if ( NULL != d_num_matched    ) { cudaFree(d_num_matched); }
        return PFAC_STATUS_CUDA_ALLOC_FAILED;
    }

    // copy input string from host to device
    cuda_status1 = cudaMemcpy(d_input_string, h_input_string, input_size, cudaMemcpyHostToDevice);
    if ( cudaSuccess != cuda_status1 ){
        cudaFree(d_input_string); 
        cudaFree(d_matched_result);
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    PFAC_status_t PFAC_status = PFAC_matchFromDevice( handle, d_input_string, input_size,
        d_matched_result ) ;

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        cudaFree(d_input_string);
        cudaFree(d_matched_result);
        return PFAC_status ;
    }

    // copy the result data from device to host
    cuda_status1 = cudaMemcpy(h_matched_result, d_matched_result, input_size*sizeof(int), cudaMemcpyDeviceToHost);
    if ( cudaSuccess != cuda_status1 ){
        cudaFree(d_input_string);
        cudaFree(d_matched_result);
        return PFAC_STATUS_INTERNAL_ERROR;
    }

    cudaFree(d_input_string);
    cudaFree(d_matched_result);

    return PFAC_STATUS_SUCCESS ;
}