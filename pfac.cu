#include "cuda_utils.h"

#include "pfac.h"
#include "pfac_match.h"

using namespace std;

const char* PFAC_getErrorString( PFAC_status_t status )
{
    static char PFAC_success_str[] = "PFAC_STATUS_SUCCESS: operation is successful" ;
    static char PFAC_alloc_failed_str[] = "PFAC_STATUS_ALLOC_FAILED: allocation fails on host memory" ;
    static char PFAC_cuda_alloc_failed_str[] = "PFAC_STATUS_CUDA_ALLOC_FAILED: allocation fails on device memory" ;
    static char PFAC_invalid_handle_str[] = "PFAC_STATUS_INVALID_HANDLE: handle is invalid (NULL)" ;
    static char PFAC_invalid_parameter_str[] = "PFAC_STATUS_INVALID_PARAMETER: parameter is invalid" ;
    static char PFAC_patterns_not_ready_str[] = "PFAC_STATUS_PATTERNS_NOT_READY: please call PFAC_readPatternFromFile() first" ;
    static char PFAC_file_open_error_str[] = "PFAC_STATUS_FILE_OPEN_ERROR: pattern file does not exist" ;
    static char PFAC_lib_not_exist_str[] = "PFAC_STATUS_LIB_NOT_EXIST: cannot find PFAC library, please check LD_LIBRARY_PATH" ;
    static char PFAC_arch_mismatch_str[] = "PFAC_STATUS_ARCH_MISMATCH: sm1.0 is not supported" ;
    static char PFAC_mutex_error[] = "PFAC_STATUS_MUTEX_ERROR: please report bugs. Workaround: choose non-texture mode.";
    static char PFAC_internal_error_str[] = "PFAC_STATUS_INTERNAL_ERROR: please report bugs" ;

    if ( PFAC_STATUS_SUCCESS == status ){
        return PFAC_success_str ;
    }
    if ( PFAC_STATUS_BASE > status ){
        return cudaGetErrorString( (cudaError_t) status ) ;
    }

    switch(status){
    case PFAC_STATUS_ALLOC_FAILED:
        return PFAC_alloc_failed_str ;
        break ;
    case PFAC_STATUS_CUDA_ALLOC_FAILED:
        return PFAC_cuda_alloc_failed_str;
        break ;
    case PFAC_STATUS_INVALID_HANDLE:
        return PFAC_invalid_handle_str ;
        break ;
    case PFAC_STATUS_INVALID_PARAMETER:
        return PFAC_invalid_parameter_str ;
        break ;
    case PFAC_STATUS_PATTERNS_NOT_READY:
        return PFAC_patterns_not_ready_str ;
        break ;
    case PFAC_STATUS_FILE_OPEN_ERROR:
        return PFAC_file_open_error_str ;
        break ;
    case PFAC_STATUS_LIB_NOT_EXIST:
        return PFAC_lib_not_exist_str ;
        break ;
    case PFAC_STATUS_ARCH_MISMATCH:
        return PFAC_arch_mismatch_str ;
        break ;
    case PFAC_STATUS_MUTEX_ERROR:
        return PFAC_mutex_error ;
        break;
    default : // PFAC_STATUS_INTERNAL_ERROR:
        return PFAC_internal_error_str ;
    }
}

 PFAC_status_t  PFAC_destroy( PFAC_handle_t handle )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    PFAC_freeResource( handle ) ;

    free( handle ) ;

    return PFAC_STATUS_SUCCESS ;
}

void  PFAC_freeResource( PFAC_handle_t handle )
{
    // resource of patterns
    if ( NULL != handle->rowPtr ){
        free( handle->rowPtr );
        handle->rowPtr = NULL ;
    }
    
    if ( NULL != handle->valPtr ){
        free( handle->valPtr );
        handle->valPtr = NULL ;
    }

    if ( NULL != handle->patternLen_table ){
        free( handle->patternLen_table ) ;
        handle->patternLen_table = NULL ;
    }
    
    if ( NULL != handle->patternID_table ){
        free( handle->patternID_table );
        handle->patternID_table = NULL ;
    }
    
    if ( NULL != handle->table_compact ){
        delete  handle->table_compact ;
        handle->table_compact = NULL ;
    }

    PFAC_freeTable( handle );
 
    handle->isPatternsReady = false ;
}

void  PFAC_freeTable( PFAC_handle_t handle )
{
    if ( NULL != handle->h_PFAC_table ){
        free( handle->h_PFAC_table ) ;
        handle->h_PFAC_table = NULL ;
    }

    if ( NULL != handle->h_hashRowPtr ){
        free( handle->h_hashRowPtr );
        handle->h_hashRowPtr = NULL ;   
    }
    
    if ( NULL != handle->h_hashValPtr ){
        free( handle->h_hashValPtr );
        handle->h_hashValPtr = NULL ;   
    }
    
    if ( NULL != handle->h_tableOfInitialState){
        free(handle->h_tableOfInitialState);
        handle->h_tableOfInitialState = NULL ; 
    }
    
    // free device resource
    if ( NULL != handle->d_PFAC_table ){
        cudaFree(handle->d_PFAC_table);
        handle->d_PFAC_table= NULL ;
    }
    
    if ( NULL != handle->d_hashRowPtr ){
        cudaFree( handle->d_hashRowPtr );
        handle->d_hashRowPtr = NULL ;
    }

    if ( NULL != handle->d_hashValPtr ){
        cudaFree( handle->d_hashValPtr );
        handle->d_hashValPtr = NULL ;   
    }
    
    if ( NULL != handle->d_tableOfInitialState ){
        cudaFree(handle->d_tableOfInitialState);
        handle->d_tableOfInitialState = NULL ;
    }   
}

PFAC_status_t  PFAC_create( PFAC_handle_t *handle )
{
    *handle = (PFAC_handle_t) malloc( sizeof(PFAC_context) ) ;

    if ( NULL == *handle ){
        return PFAC_STATUS_ALLOC_FAILED ;
    }

    memset( *handle, 0, sizeof(PFAC_context) ) ;

    // bind proper library sm_20, sm_13, sm_11 ...
    char modulepath[1+ PATH_MAX];
    void *module = NULL;

    int device ;
    cudaError_t cuda_status = cudaGetDevice( &device ) ;
    if ( cudaSuccess != cuda_status ){
        return (PFAC_status_t)cuda_status ;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    PFAC_PRINTF("major = %d, minor = %d, name=%s\n", deviceProp.major, deviceProp.minor, deviceProp.name );

    int device_no = 10*deviceProp.major + deviceProp.minor ;
//    if ( 30 == device_no ){
//        strcpy (modulepath, "libpfac_sm30.so");    
//    }else if ( 21 == device_no ){
//        strcpy (modulepath, "libpfac_sm21.so");    
//    }else if ( 20 == device_no ){
//        strcpy (modulepath, "libpfac_sm20.so");
//    }else if ( 13 == device_no ){
//        strcpy (modulepath, "libpfac_sm13.so");
//    }else if ( 12 == device_no ){
//        strcpy (modulepath, "libpfac_sm12.so");
//    }else if ( 11 == device_no ){
//        strcpy (modulepath, "libpfac_sm11.so");
//    }else{
//        return PFAC_STATUS_ARCH_MISMATCH ;
//    }
    
    (*handle)->device_no = device_no ;
    
//    PFAC_PRINTF("load module %s \n", modulepath );

    // Load the module.
//    module = dlopen (modulepath, RTLD_NOW);
//    if (!module){
//        PFAC_PRINTF("Error: modulepath(%s) cannot load module, %s\n", modulepath, dlerror() ); 
//        return PFAC_STATUS_LIB_NOT_EXIST ;
//    }

    // Find entry point of PFAC_kernel
    (*handle)->kernel_ptr = (PFAC_kernel_protoType) PFAC_kernel_timeDriven_wrapper;
    if ( NULL == (*handle)->kernel_ptr ){
        PFAC_PRINTF("Error: cannot load PFAC_kernel_timeDriven_wrapper, error = %s\n", "" );
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

//    (*handle)->kernel_space_driven_ptr = (PFAC_kernel_protoType) dlsym (module, "PFAC_kernel_spaceDriven_warpper");
//    if ( NULL == (*handle)->kernel_space_driven_ptr ){
//        PFAC_PRINTF("Error: cannot load PFAC_kernel_spaceDriven_warpper, error = %s\n", dlerror() );
//        return PFAC_STATUS_INTERNAL_ERROR ;
//    }
//    
//    // Find entry point of PFAC_reduce_kernel
//    (*handle)->reduce_kernel_ptr = (PFAC_reduce_kernel_protoType) dlsym (module, "PFAC_reduce_kernel");
//    if ( NULL == (*handle)->reduce_kernel_ptr ){
//        PFAC_PRINTF("Error: cannot load PFAC_reduce_kernel, error = %s\n", dlerror() );
//        return PFAC_STATUS_INTERNAL_ERROR ;
//    }
//
//    // Find entry point of PFAC_reduce_inplace_kernel
//    (*handle)->reduce_inplace_kernel_ptr = (PFAC_reduce_kernel_protoType) dlsym (module, "PFAC_reduce_inplace_kernel");
//    if ( NULL == (*handle)->reduce_inplace_kernel_ptr ){
//        PFAC_PRINTF("Error: cannot load PFAC_reduce_inplace_kernel, error = %s\n", dlerror() );
//        return PFAC_STATUS_INTERNAL_ERROR ;
//    }

    return PFAC_STATUS_SUCCESS ;
}