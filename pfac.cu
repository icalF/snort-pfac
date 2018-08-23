#include <cstring>

#include "cuda_utils.h"

#include "pfac.h"
#include "pfac_match.h"
#include "pfac_table.h"

static inline void ConvertCaseEx (unsigned char *d, unsigned char *s, int m)
{
    int i;
    for (i = 0; i < m; i++)
    {
        d[i] = xlatcase[s[i]];
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

    // if ( NULL != handle->h_hashRowPtr ){
    //     free( handle->h_hashRowPtr );
    //     handle->h_hashRowPtr = NULL ;   
    // }
    
    // if ( NULL != handle->h_hashValPtr ){
    //     free( handle->h_hashValPtr );
    //     handle->h_hashValPtr = NULL ;   
    // }
    
    if ( NULL != handle->h_tableOfInitialState){
        free(handle->h_tableOfInitialState);
        handle->h_tableOfInitialState = NULL ; 
    }
    
    // free device resource
    if ( NULL != handle->d_PFAC_table ){
        cudaFree(handle->d_PFAC_table);
        handle->d_PFAC_table= NULL ;
    }
    
    // if ( NULL != handle->d_hashRowPtr ){
    //     cudaFree( handle->d_hashRowPtr );
    //     handle->d_hashRowPtr = NULL ;
    // }

    // if ( NULL != handle->d_hashValPtr ){
    //     cudaFree( handle->d_hashValPtr );
    //     handle->d_hashValPtr = NULL ;   
    // }
    
    if ( NULL != handle->d_tableOfInitialState ){
        cudaFree(handle->d_tableOfInitialState);
        handle->d_tableOfInitialState = NULL ;
    }   
}

PFAC_status_t PFAC_tex_mutex_lock(PFAC_handle_t handle)
{
    // try
    // {
    //     handle->__pfac_tex_mutex.lock();
    // }
    // catch (const system_error &e)
    // {
    //     return PFAC_STATUS_MUTEX_ERROR;
    // }

    return PFAC_STATUS_SUCCESS;
}

PFAC_status_t PFAC_tex_mutex_unlock(PFAC_handle_t handle)
{
    // try
    // {
    //     handle->__pfac_tex_mutex.unlock();
    // }
    // catch (const system_error &e)
    // {
    //     return PFAC_STATUS_MUTEX_ERROR;
    // }

    return PFAC_STATUS_SUCCESS;
}

PFAC_status_t  PFAC_create( PFAC_handle_t handle )
{
    int device ;
    cudaError_t cuda_status = cudaGetDevice( &device ) ;
    if ( cudaSuccess != cuda_status ){
        return (PFAC_status_t)cuda_status ;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    PFAC_PRINTF("major = %d, minor = %d, name=%s\n", deviceProp.major, deviceProp.minor, deviceProp.name );

    int device_no = 10*deviceProp.major + deviceProp.minor ;
    handle->device_no = device_no ;

    // Find entry point of PFAC_kernel
    handle->kernel_ptr = (PFAC_kernel_protoType) PFAC_kernel_timeDriven_wrapper;
    if ( NULL == handle->kernel_ptr ){
        PFAC_PRINTF("Error: cannot load PFAC_kernel_timeDriven_wrapper, error = %s\n", "" );
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    return PFAC_STATUS_SUCCESS ;
}



PFAC_STRUCT * pfacNew (void (*userfree)(void *p),
        void (*optiontreefree)(void **p),
        void (*neg_list_free)(void **p))
{
    PFAC_handle_t handle = (PFAC_handle_t) malloc( sizeof(PFAC_STRUCT) ) ;
    if ( handle == NULL ){
        PFAC_PRINTF("Error: cannot initialize handler, error = %s\n", PFAC_getErrorString(PFAC_STATUS_ALLOC_FAILED));
        return NULL;
    }

    memset( handle, 0, sizeof(PFAC_STRUCT) ) ;

    PFAC_status_t status = PFAC_create( handle );
    if ( status != PFAC_STATUS_SUCCESS )
    {
        PFAC_PRINTF("Error: cannot initialize handler, error = %s\n", PFAC_getErrorString(status));
        return NULL;
    }

    init_xlatcase();

    handle->userfree              = userfree;
    handle->optiontreefree        = optiontreefree;
    handle->neg_list_free         = neg_list_free;

    return (PFAC_STRUCT *) handle;
}

void pfacFree ( PFAC_STRUCT * pfac )
{
    PFAC_handle_t handle = (PFAC_handle_t) pfac;
    PFAC_status_t status = PFAC_destroy( handle ) ;
    PFAC_PRINTF("%d\n", handle);
    if ( status != PFAC_STATUS_SUCCESS )
    {
        PFAC_PRINTF("Error: cannot deinitialize handler, error = %s\n", PFAC_getErrorString(status));
    }
}

int pfacAddPattern ( PFAC_STRUCT * p, unsigned char *pat, int n, int nocase,
                     int offset, int depth, int negative, void * id, int iid )
{
    PFAC_PATTERN * plist;
    plist = (PFAC_PATTERN *) calloc (1, sizeof (PFAC_PATTERN));
    plist->patrn = (uint8_t *) calloc (n, 1);
    ConvertCaseEx (plist->patrn, pat, n);
    plist->casepatrn = (uint8_t *) calloc (n, 1);
    memcpy (plist->casepatrn, pat, n);

    plist->udata = (PFAC_USERDATA *) calloc (1, sizeof (PFAC_USERDATA));
    plist->udata->ref_count = 1;
    plist->udata->id = id;

    plist->n = n;
    plist->nocase = nocase;
    plist->negative = negative;
    plist->offset = offset;
    plist->depth = depth;
    plist->iid = iid;
    plist->next = p->pfacPatterns;
    
    p->pfacPatterns = plist;
    p->numOfPatterns++;
    p->max_numOfStates += n + 1;
    return 0;
}


int pfacCompile ( PFAC_STRUCT * pfac,
        int (*build_tree)(void * id, void **existing_tree),
        int (*neg_list_func)(void *id, void **list))
{
    int max_numOfStates = ++pfac->max_numOfStates;

    // Allocate a buffer to contains all patterns
    pfac->valPtr = (char*)calloc(max_numOfStates, sizeof( char ));
    if (NULL == pfac->valPtr) {
        return PFAC_STATUS_ALLOC_FAILED;
    }

    PFAC_status_t status = PFAC_fillPatternTable((PFAC_handle_t) pfac);
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_fillPatternTable, %s\n", PFAC_getErrorString(status) );
        PFAC_freeResource( (PFAC_handle_t) pfac );
        return 0;
    }

    status = PFAC_prepareTable((PFAC_handle_t) pfac);
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_prepareTable, %s\n", PFAC_getErrorString(status) );
        PFAC_freeResource( (PFAC_handle_t) pfac );
        return 0;
    }

    return 0;
}

int pfacSearch ( PFAC_STRUCT * pfac,unsigned char * T, int n, 
        int (*Match)(void * id, void *tree, int index, void *data, void *neg_list),
        void * data, int* current_state )
{
    int *h_matched_result = (int *) malloc ( n * sizeof(int) );
    int *h_num_matched = (int *) malloc ( THREAD_BLOCK_SIZE * sizeof(int) );
    int nfound = 0;
    PFAC_handle_t handle = (PFAC_handle_t) pfac;

    PFAC_status_t status = PFAC_matchFromHost( handle, (char *) T, n, h_matched_result, h_num_matched ) ;

    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_matchFromHost, %s\n", PFAC_getErrorString(status) );
        return 0;
    }

    #pragma omp parallel for reduction (+:nfound)
    for (int i = 0; i < THREAD_BLOCK_SIZE; ++i)
    {
        nfound += h_num_matched[i];
    }

    return nfound;
}

int pfacPrintDetailInfo(PFAC_STRUCT * p)
{
    if(p)
        p = p;
    return 0;
}

int pfacPrintSummaryInfo(void)
{
    // SPFAC_STRUCT2 * p = &summary.spfac;

    // if( !summary.num_states )
    //     return;

    // PFAC_PRINTF("+--[Pattern Matcher:Aho-Corasick Summary]----------------------\n");
    // PFAC_PRINTF("| Alphabet Size    : %d Chars\n",p->spfacAlphabetSize);
    // PFAC_PRINTF("| Sizeof State     : %d bytes\n",sizeof(acstate_t));
    // PFAC_PRINTF("| Storage Format   : %s \n",sf[ p->spfacFormat ]);
    // PFAC_PRINTF("| Num States       : %d\n",summary.num_states);
    // PFAC_PRINTF("| Num Transitions  : %d\n",summary.num_transitions);
    // PFAC_PRINTF("| State Density    : %.1f%%\n",100.0*(double)summary.num_transitions/(summary.num_states*p->spfacAlphabetSize));
    // PFAC_PRINTF("| Finite Automatum : %s\n", fsa[p->spfacFSA]);
    // if( max_memory < 1024*1024 )
    //     PFAC_PRINTF("| Memory           : %.2fKbytes\n", (float)max_memory/1024 );
    // else
    //     PFAC_PRINTF("| Memory           : %.2fMbytes\n", (float)max_memory/(1024*1024) );
    // PFAC_PRINTF("+-------------------------------------------------------------\n");

    return 0;
}
