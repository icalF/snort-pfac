#include "cuda_utils.h"
#include "pfac.h"

#include <cassert>
#include <vector>
#include <algorithm>

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

int lookup(vector< vector<TableEle> > &table, const int state, const int ch)
{
	if (state >= table.size()) { return TRAP_STATE; }
	for (int j = 0; j < table[state].size(); j++) {
		TableEle ele = table[state][j];
		if (ch == ele.ch) {
			return ele.nextState;
		}
	}
	return TRAP_STATE;
}

void printString( char *s, const int n, FILE* fp )
{
    fprintf(fp,"%c", '\"');
    for( int i = 0 ; i < n ; i++){
        int ch = (unsigned char) s[i] ;
        if ( (32 <= ch) && (126 >= ch) ){
            fprintf(fp,"%c", ch );
        }else{
            fprintf(fp,"%2.2x", ch );
        }        
    }
    fprintf(fp,"%c", '\"');
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

//    if ( handle->isPatternsReady ){
//        // free previous patterns, including transition tables in host and device memory
//        PFAC_freeResource( handle );
//    }

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
        
    return PFAC_status ;
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
PFAC_status_t parsePatternFile(char *patternfilename,
	char ***rowPtr, char **valPtr, int **patternID_table_ptr, int **patternLen_table_ptr,
	int *max_state_num_ptr, int *pattern_num_ptr)
{
	if (NULL == patternfilename) {
		return PFAC_STATUS_INVALID_PARAMETER;
	}
	if (NULL == rowPtr) {
		return PFAC_STATUS_INVALID_PARAMETER;
	}
	if (NULL == valPtr) {
		return PFAC_STATUS_INVALID_PARAMETER;
	}
	if (NULL == patternID_table_ptr) {
		return PFAC_STATUS_INVALID_PARAMETER;
	}
	if (NULL == patternLen_table_ptr) {
		return PFAC_STATUS_INVALID_PARAMETER;
	}
	if (NULL == max_state_num_ptr) {
		return PFAC_STATUS_INVALID_PARAMETER;
	}
	if (NULL == pattern_num_ptr) {
		return PFAC_STATUS_INVALID_PARAMETER;
	}

	FILE* fpin = fopen(patternfilename, "rb");
	if (fpin == NULL) {
		PFAC_PRINTF("Error: Open pattern file %s failed.", patternfilename);
		return PFAC_STATUS_FILE_OPEN_ERROR;
	}

	// step 1: find size of the file
	// obtain file size
	fseek(fpin, 0, SEEK_END);
	int file_size = ftell(fpin);
	rewind(fpin);

	// step 2: allocate a buffer to contains all patterns
	*valPtr = (char*)malloc(sizeof(char)*file_size);
	if (NULL == *valPtr) {
		return PFAC_STATUS_ALLOC_FAILED;
	}

	// copy the file into the buffer
	file_size = fread(*valPtr, 1, file_size, fpin);
	fclose(fpin);

	char *buffer = *valPtr;
	vector< struct patternEle > rowIdxArray;
	vector<int>  patternLenArray;
	int len;

	struct patternEle pEle;

	pEle.patternString = buffer;
	pEle.patternID = 1;

	rowIdxArray.push_back(pEle);
	len = 0;
	for (int i = 0; i < file_size; i++) {
		if ('\n' == buffer[i]) {
			if ((i > 0) && ('\n' != buffer[i - 1])) { // non-empty line
				patternLenArray.push_back(len);
				pEle.patternString = buffer + i + 1; // start of next pattern
				pEle.patternID = rowIdxArray.size() + 1; // ID of next pattern
				rowIdxArray.push_back(pEle);
			}
			len = 0;
		}
		else {
			len++;
		}
	}

	*pattern_num_ptr = rowIdxArray.size() - 1;
	*max_state_num_ptr = file_size + 1;

	// rowIdxArray.size()-1 = number of patterns
	// sort patterns by lexicographic order
	sort(rowIdxArray.begin(), rowIdxArray.begin() + *pattern_num_ptr, pattern_cmp_functor());

	*rowPtr = (char**)malloc(sizeof(char*)*rowIdxArray.size());
	*patternID_table_ptr = (int*)malloc(sizeof(int)*rowIdxArray.size());
	// suppose there are k patterns, then size of patternLen_table is k+1
	// because patternLen_table[0] is useless, valid data starts from
	// patternLen_table[1], up to patternLen_table[k]
	*patternLen_table_ptr = (int*)malloc(sizeof(int)*rowIdxArray.size());
	if ((NULL == *rowPtr) ||
		(NULL == *patternID_table_ptr) ||
		(NULL == *patternLen_table_ptr))
	{
		return PFAC_STATUS_ALLOC_FAILED;
	}

	// step 5: compute f(final state) = patternID
	for (int i = 0; i < (rowIdxArray.size() - 1); i++) {
		(*rowPtr)[i] = rowIdxArray[i].patternString;
		(*patternID_table_ptr)[i] = rowIdxArray[i].patternID; // pattern number starts from 1
	}

	// although patternLen_table[0] is useless, in order to avoid errors from valgrind
	// we need to initialize patternLen_table[0]
	(*patternLen_table_ptr)[0] = 0;
	for (int i = 0; i < (rowIdxArray.size() - 1); i++) {
		// pattern (*rowPtr)[i] is terminated by character '\n'
		// pattern ID starts from 1, so patternID = i+1
		(*patternLen_table_ptr)[i + 1] = patternLenArray[i];
	}

	return PFAC_STATUS_SUCCESS;
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
	vector< vector<TableEle> > &PFAC_table)
{
	int state;
	int state_num;

	PFAC_table.clear();
	PFAC_table.reserve(max_state_num);
	vector< TableEle > empty_row;
	for (int i = 0; i < max_state_num; i++) {
		PFAC_table.push_back(empty_row);
	}

	PFAC_PRINTF("initial state : %d\n", initial_state);

	state = initial_state; // state is current state
	//state_num = initial_state + 1; // state_num: usable state
	state_num = baseOfUsableStateID;

	for (int p_idx = 0; p_idx < pattern_num; p_idx++) {
		char *pos = (char*)rowPtr[p_idx];
		int  patternID = patternID_table[p_idx];
		int  len = patternLen_table[patternID];

		/*
				printf("pid = %d, length = %d, ", patternID, len );
				printStringEndNewLine( pos, stdout );
				printf("\n");
		*/

		for (int offset = 0; offset < len; offset++) {
			int ch = (unsigned char)pos[offset];
			assert('\n' != ch);

			if ((len - 1) == offset) { // finish reading a pattern
				TableEle ele;
				ele.ch = ch;
				ele.nextState = patternID; // patternID is id of final state
				PFAC_table[state].push_back(ele); //PFAC_table[ PFAC_TABLE_MAP(state,ch) ] = patternID; 
				state = initial_state;
			}
			else {
				int nextState = lookup(PFAC_table, state, ch);
				if (TRAP_STATE == nextState) {
					TableEle ele;
					ele.ch = ch;
					ele.nextState = state_num;
					PFAC_table[state].push_back(ele); // PFAC_table[PFAC_TABLE_MAP(state,ch)] = state_num;
					state = state_num; // go to next state
					state_num = state_num + 1; // next available state
				}
				else {
					// match prefix of previous pattern
					// state = PFAC_table[PFAC_TABLE_MAP(state,ch)]; // go to next state
					state = nextState;
				}
			}

			if (state_num > max_state_num) {
				PFAC_PRINTF("Error: State number overflow, state no=%d, max_state_num=%d\n", state_num, max_state_num);
				return PFAC_STATUS_INTERNAL_ERROR;
			}
		}  // while
	}  // for each pattern

	PFAC_PRINTF("The number of state is %d\n", state_num);

	*state_num_ptr = state_num;

	return PFAC_STATUS_SUCCESS;
}

#define  PFAC_TABLE_MAP( i , j )   (i)*CHAR_SET + (j)

PFAC_status_t  PFAC_dumpTransitionTable( PFAC_handle_t handle, FILE *fp )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( NULL == fp ){
        fp = stdout ;
    }
    int state_num = handle->numOfStates ;
    int num_finalState = handle->numOfFinalStates ;
    int initial_state = handle->initial_state ;
    int *patternLen_table = handle->patternLen_table ;
    int *patternID_table = handle->patternID_table ;

    fprintf(fp,"# Transition table: number of states = %d, initial state = %d\n", state_num, initial_state );
    fprintf(fp,"# (current state, input character) -> next state \n");

    for(int state = 0 ; state < state_num ; state++ ){
        for(int j = 0 ; j < (int)(*(handle->table_compact))[state].size(); j++){
            TableEle ele = (*(handle->table_compact))[state][j];
            int ch = ele.ch ;
            int nextState = ele.nextState;
            if ( TRAP_STATE != nextState ){
                if ( (32 <= ch) && (126 >= ch) ){
                    fprintf(fp,"(%4d,%4c) -> %d \n", state, ch, nextState );
                }else{
                    fprintf(fp,"(%4d,%4.2x) -> %d \n", state, ch, nextState );
                }
            }
        }	
    }

    vector< char* > origin_patterns(num_finalState) ;
    for( int i = 0 ; i < num_finalState ; i++){
        char *pos = (handle->rowPtr)[i] ;
        int patternID = patternID_table[i] ;
        origin_patterns[patternID-1] = pos ;
    }

    fprintf(fp,"# Output table: number of final states = %d\n", num_finalState );
    fprintf(fp,"# [final state] [matched pattern ID] [pattern length] [pattern(string literal)] \n");

    for( int state = 1 ; state <= num_finalState ; state++){
        int patternID = state;
        int len = patternLen_table[patternID];
        if ( 0 != patternID ){
            fprintf(fp, "%5d %5d %5d    ", state, patternID, len );
            char *pos = origin_patterns[patternID-1] ;
            //printStringEndNewLine( pos, fp );
            printString( pos, len, fp );
            fprintf(fp, "\n" );
        }else{
            return PFAC_STATUS_INTERNAL_ERROR ;
        }
    }

    return PFAC_STATUS_SUCCESS ;
}

/*
 *  suppose N = number of states
 *          C = number of character set = 256
 *
 *  TIME-DRIVEN:
 *     allocate a explicit 2-D table with N*C integers.
 *     host: 
 *          h_PFAC_table
 *     device: 
 *          d_PFAC_table
 *
 *  SPACE-DRIVEN:
 *     allocate a hash table (hashRowPtr, hashValPtr)
 *     host:
 *          h_hashRowPtr
 *          h_hashValPtr
 *          h_tableOfInitialState
 *     device:
 *          d_hashRowPtr
 *          d_hashValPtr
 *          d_tableOfInitialState         
 */
PFAC_status_t  PFAC_bindTable( PFAC_handle_t handle )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    
    PFAC_status_t PFAC_status ;
    PFAC_status = PFAC_create2DTable(handle);
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){  	
        PFAC_PRINTF("Error: cannot create transistion table \n");	
        return PFAC_status ;
    }

    return PFAC_STATUS_SUCCESS ;
}

 
PFAC_status_t  PFAC_create2DTable( PFAC_handle_t handle )
{
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }	
    
    /* perfMode is PFAC_TIME_DRIVEN, we don't need to allocate 2-D table again */
    if ( NULL != handle->d_PFAC_table ){
        return PFAC_STATUS_SUCCESS ;
    }	
    
    const int numOfStates = handle->numOfStates ;

    handle->numOfTableEntry = CHAR_SET*numOfStates ; 
    handle->sizeOfTableEntry = sizeof(int) ; 
    handle->sizeOfTableInBytes = (handle->numOfTableEntry) * (handle->sizeOfTableEntry) ; 

#define  PFAC_TABLE_MAP( i , j )   (i)*CHAR_SET + (j)    

    if ( NULL == handle->h_PFAC_table){    
        handle->h_PFAC_table = (int*) malloc( handle->sizeOfTableInBytes ) ;
        if ( NULL == handle->h_PFAC_table ){
            return PFAC_STATUS_ALLOC_FAILED ;
        }
    
        // initialize PFAC table to TRAP_STATE
        for (int i = 0; i < numOfStates ; i++) {
            for (int j = 0; j < CHAR_SET; j++) {
                (handle->h_PFAC_table)[ PFAC_TABLE_MAP( i , j ) ] = TRAP_STATE ;
            }
        }
        for(int i = 0 ; i < numOfStates ; i++ ){
            for(int j = 0 ; j < (int)(*(handle->table_compact))[i].size(); j++){
                TableEle ele = (*(handle->table_compact))[i][j];
                (handle->h_PFAC_table)[ PFAC_TABLE_MAP( i , ele.ch ) ] = ele.nextState;  	
            }
        }
    }

    cudaError_t cuda_status = cudaMalloc((void **) &handle->d_PFAC_table, handle->sizeOfTableInBytes );
    if ( cudaSuccess != cuda_status ){
        free(handle->h_PFAC_table);
        handle->h_PFAC_table = NULL ;
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }

    cuda_status = cudaMemcpy(handle->d_PFAC_table, handle->h_PFAC_table,
        handle->sizeOfTableInBytes, cudaMemcpyHostToDevice);
    if ( cudaSuccess != cuda_status ){
        free(handle->h_PFAC_table);
        handle->h_PFAC_table = NULL ;
        cudaFree(handle->d_PFAC_table);
        handle->d_PFAC_table = NULL;    	
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
    
    return PFAC_STATUS_SUCCESS ;
}


inline void correctTextureMode(PFAC_handle_t handle)
{		
    PFAC_PRINTF("handle->textureMode = %d\n",handle->textureMode );	
    /* maximum width for a 1D texture reference is independent of type */
    if ( PFAC_AUTOMATIC == handle->textureMode ){
        if ( handle->numOfTableEntry < MAXIMUM_WIDTH_1DTEX ){ 
            PFAC_PRINTF("reset to tex on, handle->numOfTableEntry =%d < %d\n",handle->numOfTableEntry, MAXIMUM_WIDTH_1DTEX);         	
            handle->textureMode = PFAC_TEXTURE_ON ;
        }else{
            PFAC_PRINTF("reset to tex off, handle->numOfTableEntry =%d > %d\n",handle->numOfTableEntry, MAXIMUM_WIDTH_1DTEX); 
            handle->textureMode = PFAC_TEXTURE_OFF ;
        }
    }
}

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

    // n_hat = number of integers of input string
    int n_hat = (input_size + sizeof(int)-1)/sizeof(int) ;

    // allocate memory for input string and result
    // basic unit of d_input_string is integer
	PFAC_PRINTF("Meong");
    cudaError_t cuda_status1 = cudaMalloc((void **) &d_input_string,        n_hat*sizeof(int) );
    cudaError_t cuda_status2 = cudaMalloc((void **) &d_matched_result, input_size*sizeof(int) );
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) ){
    	  if ( NULL != d_input_string   ) { cudaFree(d_input_string); }
    	  if ( NULL != d_matched_result ) { cudaFree(d_matched_result); }
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

void initiate( char ***rowPtr, 
    char **valPtr, int **patternID_table_ptr, int **patternLen_table_ptr,
    int *max_state_num_ptr, int *pattern_num_ptr, int *state_num_ptr )
{
//	rowPtr = ;
//	state_num_ptr = (int*)malloc(sizeof(int));
//	filename = (char*)malloc(sizeof(char));
}

void destroy( char ***rowPtr, 
    char **valPtr, int **patternID_table_ptr, int **patternLen_table_ptr,
    int *max_state_num_ptr, int *pattern_num_ptr, int *state_num_ptr )
{
	free(rowPtr);
	free(valPtr);
	free(patternID_table_ptr);
	free(patternLen_table_ptr);
	free(max_state_num_ptr);
	free(pattern_num_ptr);
	free(state_num_ptr);
}

int main(int argc, char **argv)
{
	char dumpTableFile[] = "table.txt";
	char inputFile[] = "test/data/example_input";
	char patternFile[] = "test/pattern/example_pattern";
	PFAC_handle_t handle;
	PFAC_status_t PFAC_status;
	int input_size;
	char *h_inputString = NULL;
	int  *h_matched_result = NULL;

	// step 1: create PFAC handle 
    PFAC_status = PFAC_create( &handle ) ;
    assert( PFAC_STATUS_SUCCESS == PFAC_status );

	// step 2: read patterns and dump transition table 
	PFAC_status = PFAC_readPatternFromFile( handle, patternFile) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to read pattern from file, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }

	// dump transition table 
//	FILE *table_fp = fopen(dumpTableFile, "w");
//	assert(NULL != table_fp);
//	PFAC_status = PFAC_dumpTransitionTable( handle, table_fp );
//	fclose(table_fp);
//	if (PFAC_STATUS_SUCCESS != PFAC_status) {
//		printf("Error: fails to dump transition table, %s\n", PFAC_getErrorString(PFAC_status));
//		exit(1);
//	}

	//step 3: prepare input stream
	FILE* fpin = fopen(inputFile, "rb");
	assert(NULL != fpin);

	// obtain file size
	fseek(fpin, 0, SEEK_END);
	input_size = ftell(fpin);
	rewind(fpin);

	// allocate memory to contain the whole file
	h_inputString = (char *)malloc(sizeof(char)*input_size);
	assert(NULL != h_inputString);

	h_matched_result = (int *)malloc(sizeof(int)*input_size);
	assert(NULL != h_matched_result);
	memset(h_matched_result, 0, sizeof(int)*input_size);

	// copy the file into the buffer
	input_size = fread(h_inputString, 1, input_size, fpin);
	fclose(fpin);

	// step 4: run PFAC on GPU           
    PFAC_status = PFAC_matchFromHost( handle, h_inputString, input_size, h_matched_result ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to PFAC_matchFromHost, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }     

	// step 5: output matched result
    for (int i = 0; i < input_size; i++) {
        if (h_matched_result[i] != 0) {
            printf("At position %4d, match pattern %d\n", i, h_matched_result[i]);
        }
    }

    PFAC_status = PFAC_destroy( handle ) ;
    assert( PFAC_STATUS_SUCCESS == PFAC_status );

	free(h_inputString);
	free(h_matched_result);

	return 0;
}