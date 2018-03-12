#include "cuda_utils.h"
#include "pfac.h"

#include <cassert>
#include <vector>
#include <pthread>

using namespace std;

/*
 *  CUDA 4.0 can supports one host thread to multiple GPU contexts.
 *  PFAC library still binds one PFAC handle to one GPU context.
 *
 *  consider followin example
 *  ----------------------------------------------------------------------
 *  cudaSetDevice(0);
 *  PFAC_create( PFAC_handle0 );
 *  PFAC_readPatternFromFile( PFAC_handle0, pattern_file )
 *  cudaSetDevice(1);
 *  PFAC_matchFromHost( PFAC_handle0, h_input_string, input_size, h_matched_result )
 *  ----------------------------------------------------------------------
 *
 *  Then PFAC library does not work because transition table of DFA is in GPU0 
 *  but d_input_string and d_matched_result are in GPU1.
 *  You can create two PFAC handles corresponding to different GPUs.
 *  ----------------------------------------------------------------------
 *  cudaSetDevice(0);
 *  PFAC_create( PFAC_handle0 );
 *  PFAC_readPatternFromFile( PFAC_handle0, pattern_file )
 *  cudaSetDevice(1);
 *  PFAC_create( PFAC_handle1 );
 *  PFAC_readPatternFromFile( PFAC_handle1, pattern_file ) 
 *  cudaSetDevice(0);
 *  PFAC_matchFromHost( PFAC_handle0, h_input_string, input_size, h_matched_result )
 *  cudaSetDevice(1);
 *  PFAC_matchFromHost( PFAC_handle1, h_input_string, input_size, h_matched_result ) 
 *  ---------------------------------------------------------------------- 
 *    
 */
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
    if ( 30 == device_no ){
        strcpy (modulepath, "libpfac_sm30.so");    
    }else if ( 21 == device_no ){
        strcpy (modulepath, "libpfac_sm21.so");    
    }else if ( 20 == device_no ){
        strcpy (modulepath, "libpfac_sm20.so");
    }else if ( 13 == device_no ){
        strcpy (modulepath, "libpfac_sm13.so");
    }else if ( 12 == device_no ){
        strcpy (modulepath, "libpfac_sm12.so");
    }else if ( 11 == device_no ){
        strcpy (modulepath, "libpfac_sm11.so");
    }else{
        return PFAC_STATUS_ARCH_MISMATCH ;
    }
    
    (*handle)->device_no = device_no ;
    
    PFAC_PRINTF("load module %s \n", modulepath );

    // Load the module.
    // module = dlopen (modulepath, RTLD_NOW);
    // if (!module){
    //     PFAC_PRINTF("Error: modulepath(%s) cannot load module, %s\n", modulepath, dlerror() ); 
    //     return PFAC_STATUS_LIB_NOT_EXIST ;
    // }

    // Find entry point of PFAC_kernel
    (*handle)->kernel_time_driven_ptr = (PFAC_kernel_protoType) PFAC_kernel_timeDriven_wrapper;
    if ( NULL == (*handle)->kernel_time_driven_ptr ){
        PFAC_PRINTF("Error: cannot load PFAC_kernel_timeDriven_warpper, error = %s\n", dlerror() );
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
    
    // (*handle)->kernel_space_driven_ptr = (PFAC_kernel_protoType) dlsym (module, "PFAC_kernel_spaceDriven_warpper");
    // if ( NULL == (*handle)->kernel_space_driven_ptr ){
    //     PFAC_PRINTF("Error: cannot load PFAC_kernel_spaceDriven_warpper, error = %s\n", dlerror() );
    //     return PFAC_STATUS_INTERNAL_ERROR ;
    // }
    
    // Find entry point of PFAC_reduce_kernel
    // (*handle)->reduce_kernel_ptr = (PFAC_reduce_kernel_protoType) dlsym (module, "PFAC_reduce_kernel");
    // if ( NULL == (*handle)->reduce_kernel_ptr ){
    //     PFAC_PRINTF("Error: cannot load PFAC_reduce_kernel, error = %s\n", dlerror() );
    //     return PFAC_STATUS_INTERNAL_ERROR ;
    // }

    // Find entry point of PFAC_reduce_inplace_kernel
    // (*handle)->reduce_inplace_kernel_ptr = (PFAC_reduce_kernel_protoType) dlsym (module, "PFAC_reduce_inplace_kernel");
    // if ( NULL == (*handle)->reduce_inplace_kernel_ptr ){
    //     PFAC_PRINTF("Error: cannot load PFAC_reduce_inplace_kernel, error = %s\n", dlerror() );
    //     return PFAC_STATUS_INTERNAL_ERROR ;
    // }

    return PFAC_STATUS_SUCCESS ;
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
        delete 	handle->table_compact ;
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

/* wrapper for pthread_mutex_lock and pthread_mutex_unlock */
pthread_mutex_t  __pfac_tex_mutex = PTHREAD_MUTEX_INITIALIZER;
PFAC_status_t PFAC_tex_mutex_lock( void )
{
    int flag = pthread_mutex_lock( &__pfac_tex_mutex);
    if ( flag ){
        return PFAC_STATUS_MUTEX_ERROR;
    }else{
        return PFAC_STATUS_SUCCESS;
    }
}

PFAC_status_t PFAC_tex_mutex_unlock( void )
{
    int flag = pthread_mutex_unlock( &__pfac_tex_mutex);
    if ( flag ){
        return PFAC_STATUS_MUTEX_ERROR;
    }else{
        return PFAC_STATUS_SUCCESS;
    }
}

int lookup(vector< vector<TableEle> > &table, const int state, const int ch )
{
    if (state >= table.size() ) { return TRAP_STATE ;}
    for(int j = 0 ; j < table[state].size() ; j++){
        TableEle ele = table[state][j];
        if ( ch == ele.ch ){
            return ele.nextState ;	
        }	
    }
    return TRAP_STATE ;
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

    if ( handle->isPatternsReady ){
        // free previous patterns, including transition tables in host and device memory
        PFAC_freeResource( handle );
    }

    if ( FILENAME_LEN > strlen(filename) ){
        strcpy( handle->patternFile, filename ) ;
    }else{
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    PFAC_status_t PFAC_status = parsePatternFile( filename,
        &handle->rowPtr, &handle->valPtr, &handle->patternID_table, &handle->patternLen_table,
        &handle->max_numOfStates, &handle->numOfPatterns ) ;

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        PFAC_freeResource( handle );
        return PFAC_status ;
    }

    int pattern_num = handle->numOfPatterns ;
    
    // compute maximum pattern length
    handle->maxPatternLen = 0 ;
    for(int i = 1 ; i <= pattern_num ; i++ ){
        if ( handle->maxPatternLen < (handle->patternLen_table)[i] ){
            handle->maxPatternLen = (handle->patternLen_table)[i];
        }
    }

    handle->initial_state  = handle->numOfPatterns + 1 ;
    handle->numOfFinalStates = handle->numOfPatterns ;

    // step 2: create PFAC table
    handle->table_compact = new vector< vector<TableEle> > ;
    if ( NULL == handle->table_compact ){
        PFAC_freeResource( handle );
        return PFAC_STATUS_ALLOC_FAILED ;
    }
    
    int baseOfUsableStateID = handle->initial_state + 1 ; // assume initial_state = handle->numOfFinalStates + 1
    PFAC_status = create_PFACTable_spaceDriven((const char**)handle->rowPtr,
        (const int*)handle->patternLen_table, (const int*)handle->patternID_table,
        handle->max_numOfStates, handle->numOfPatterns, handle->initial_state, baseOfUsableStateID, 
        &handle->numOfStates, *(handle->table_compact) );

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        PFAC_freeResource( handle );
        return PFAC_status ;
    }
    
    // compute numOfLeaves = number of leaf nodes
    // leaf node only appears in the final states
    handle->numOfLeaves = 0 ;
    for(int i = 1 ; i <= handle->numOfPatterns ; i++ ){
        // s0 is useless, so ignore s0
        if ( 0 == (*handle->table_compact)[i].size() ){
            handle->numOfLeaves ++ ;	
        }
    }
    
    // step 3: copy data to device memory
    handle->isPatternsReady = true ;

    PFAC_status = PFAC_bindTable( handle ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status){
         PFAC_freeResource( handle );
         handle->isPatternsReady = false ;
         return PFAC_status ;
    }
        
    return PFAC_STATUS_SUCCESS ;
}

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
PFAC_status_t parsePatternFile( char *patternfilename,
    char ***rowPtr, char **valPtr, int **patternID_table_ptr, int **patternLen_table_ptr,
    int *max_state_num_ptr, int *pattern_num_ptr )
{
    if ( NULL == patternfilename ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == rowPtr ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == valPtr ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == patternID_table_ptr ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == patternLen_table_ptr ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == max_state_num_ptr ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == pattern_num_ptr ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    FILE* fpin = fopen(patternfilename, "rb");
    if (fpin == NULL) {
        PFAC_PRINTF("Error: Open pattern file %s failed.", patternfilename );
        return PFAC_STATUS_FILE_OPEN_ERROR ;
    }

    // step 1: find size of the file
    // obtain file size
    fseek (fpin , 0 , SEEK_END);
    int file_size = ftell (fpin);
    rewind (fpin);

    // step 2: allocate a buffer to contains all patterns
    *valPtr = (char*)malloc(sizeof(char)*file_size ) ;
    if ( NULL == *valPtr ){
        return PFAC_STATUS_ALLOC_FAILED ;
    }

    // copy the file into the buffer
    file_size = fread (*valPtr, 1, file_size, fpin);
    fclose(fpin);

    char *buffer = *valPtr ;
    vector< struct patternEle > rowIdxArray ;
    vector<int>  patternLenArray ;
    int len ;

    struct patternEle pEle;

    pEle.patternString = buffer ;
    pEle.patternID = 1 ;
 
    rowIdxArray.push_back(pEle) ;
    len = 0 ;
    for( int i = 0 ; i < file_size ; i++){
        if ( '\n' == buffer[i] ){
            if ( (i > 0) && ('\n' != buffer[i-1]) ){ // non-empty line
                patternLenArray.push_back(len);
                pEle.patternString = buffer + i + 1; // start of next pattern
                pEle.patternID = rowIdxArray.size()+1; // ID of next pattern
                rowIdxArray.push_back(pEle) ;
            }
            len = 0 ;
        }else{
            len++ ;
        }
    }

    *pattern_num_ptr = rowIdxArray.size() - 1 ;
    *max_state_num_ptr = file_size + 1 ;

    // rowIdxArray.size()-1 = number of patterns
    // sort patterns by lexicographic order
    sort(rowIdxArray.begin(), rowIdxArray.begin()+*pattern_num_ptr, pattern_cmp_functor() ) ;

    *rowPtr = (char**) malloc( sizeof(char*)*rowIdxArray.size() ) ;
    *patternID_table_ptr = (int*) malloc( sizeof(int)*rowIdxArray.size() ) ;
    // suppose there are k patterns, then size of patternLen_table is k+1
    // because patternLen_table[0] is useless, valid data starts from
    // patternLen_table[1], up to patternLen_table[k]
    *patternLen_table_ptr = (int*) malloc( sizeof(int)*rowIdxArray.size() ) ;
    if ( ( NULL == *rowPtr ) ||
         ( NULL == *patternID_table_ptr ) ||
         ( NULL == *patternLen_table_ptr ) )
    {
        return PFAC_STATUS_ALLOC_FAILED ;
    }

    // step 5: compute f(final state) = patternID
    for( int i = 0 ; i < (rowIdxArray.size()-1) ; i++){
        (*rowPtr)[i] = rowIdxArray[i].patternString ;
        (*patternID_table_ptr)[i] = rowIdxArray[i].patternID ; // pattern number starts from 1
    }

    // although patternLen_table[0] is useless, in order to avoid errors from valgrind
    // we need to initialize patternLen_table[0]
    (*patternLen_table_ptr)[0] = 0 ;
    for( int i = 0 ; i < (rowIdxArray.size()-1) ; i++){
        // pattern (*rowPtr)[i] is terminated by character '\n'
        // pattern ID starts from 1, so patternID = i+1
        (*patternLen_table_ptr)[i+1] = patternLenArray[i] ;
    }

    return PFAC_STATUS_SUCCESS ;
}

/*
 *  Given k = pattern_number patterns in rowPtr[0:k-1] with lexicographic order and
 *  patternLen_table[1:k], patternID_table[0:k-1]
 *
 *  user specified a initial state "initial_state",
 *  construct
 *  (1) PFAC_table: DFA of PFAC with k final states labeled from 1:k
 *
 *  WARNING: initial_state = k+1
 */
PFAC_status_t create_PFACTable_spaceDriven(const char** rowPtr, const int *patternLen_table, const int *patternID_table,
    const int max_state_num,
    const int pattern_num, const int initial_state, const int baseOfUsableStateID, 
    int *state_num_ptr,
    vector< vector<TableEle> > &PFAC_table )
{
    int state ;
    int state_num ;

    PFAC_table.clear();
    PFAC_table.reserve( max_state_num );
    vector< TableEle > empty_row ;
    for(int i = 0 ; i < max_state_num ; i++){   
        PFAC_table.push_back( empty_row );
    }
    
    PFAC_PRINTF("initial state : %d\n", initial_state);

    state = initial_state; // state is current state
    //state_num = initial_state + 1; // state_num: usable state
    state_num = baseOfUsableStateID ;

    for ( int p_idx = 0 ; p_idx < pattern_num ; p_idx++ ) {
        char *pos = (char*) rowPtr[p_idx] ;
        int  patternID = patternID_table[p_idx];
        int  len = patternLen_table[patternID] ;

/*
        printf("pid = %d, length = %d, ", patternID, len );
        printStringEndNewLine( pos, stdout );
        printf("\n");
*/

        for( int offset = 0 ; offset < len  ; offset++ ){
            int ch = (unsigned char) pos[offset];
            assert( '\n' != ch ) ;

            if ( (len-1) == offset ) { // finish reading a pattern
                TableEle ele ;
                ele.ch = ch ;
                ele.nextState = patternID ; // patternID is id of final state
                PFAC_table[state].push_back(ele); //PFAC_table[ PFAC_TABLE_MAP(state,ch) ] = patternID; 
                state = initial_state;
            }
            else {
                int nextState = lookup(PFAC_table, state, ch );
                if (TRAP_STATE == nextState ) {
                    TableEle ele ;
                    ele.ch = ch ;
                    ele.nextState = state_num ;
                    PFAC_table[state].push_back(ele); // PFAC_table[PFAC_TABLE_MAP(state,ch)] = state_num;
                    state = state_num; // go to next state
                    state_num = state_num + 1; // next available state
                }
                else {
                    // match prefix of previous pattern
                    // state = PFAC_table[PFAC_TABLE_MAP(state,ch)]; // go to next state
                    state = nextState ;
                }
            }

            if (state_num > max_state_num) {
                PFAC_PRINTF("Error: State number overflow, state no=%d, max_state_num=%d\n", state_num, max_state_num );
                return PFAC_STATUS_INTERNAL_ERROR ;
            }
        }  // while
    }  // for each pattern

    PFAC_PRINTF("The number of state is %d\n", state_num);

    *state_num_ptr = state_num ;

    return PFAC_STATUS_SUCCESS ;
}

int main (int argc, char **argv) 
{
    PFAC_handle_t handle ;
    PFAC_status_t PFAC_status ;
    int input_size ;    
    char *h_inputString = NULL ;
    int  *h_matched_result = NULL ;

    return 0;
}