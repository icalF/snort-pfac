#ifndef PFAC_H_
#define PFAC_H_

#include <cstdio>
#include <vector>
#include <mutex>

// #ifdef __cplusplus
// extern "C" {
// #endif   // __cplusplus

/* This is missing from very old Linux libc. */
#ifndef RTLD_NOW
# define RTLD_NOW 2
#endif

#include <limits.h>
#ifndef PATH_MAX
# define PATH_MAX 255
#endif


#define THREAD_BLOCK_EXP   (8)
#define MAX_DIM_GRID_EXP   (20)
#define EXTRA_SIZE_PER_TB  (128)
#define MAX_DIM_GRID       (1 << MAX_DIM_GRID_EXP)
#define THREAD_BLOCK_SIZE  (1 << THREAD_BLOCK_EXP)


#if THREAD_BLOCK_SIZE != 256 
    #error THREAD_BLOCK_SIZE != 256 
#endif


using namespace std;

/*
 * debug mode:  PFAC_PRINTF( ... ) printf( __VA_ARGS__ )
 * release mode:  PFAC_PRINTF( ... )
 */
#define DEBUG
#ifndef DEBUG 
# define PFAC_PRINTF(...)
#else
# define PFAC_PRINTF(...) printf( __VA_ARGS__ ); /*fflush(stdout);*/
#endif

/* maximum width for a 1D texture reference bound to linear memory, independent of size of element*/
#define  MAXIMUM_WIDTH_1DTEX    (1 << 27)

/*
 *  The purpose of PFAC_STATUS_BASE is to separate CUDA error code and PFAC error code
 *  but align PFAC_STATUS_SUCCESS to cudaSuccess.
 *
 *  cudaError_enum is defined in /usr/local/cuda/include/cuda.h
 *  The last one is
 *      CUDA_ERROR_UNKNOWN                        = 999
 *
 *  That is why PFAC_STATUS_BASE = 10000 > 999
 *
 *  However now we regard all CUDA non-allocation error as PFAC_STATUS_INTERNAL_ERROR,
 *  PFAC_STATUS_BASE may be removed in the future 
 */
typedef enum {
    PFAC_STATUS_SUCCESS = 0 ,
    PFAC_STATUS_BASE = 10000, 
    PFAC_STATUS_ALLOC_FAILED,
    PFAC_STATUS_CUDA_ALLOC_FAILED,    
    PFAC_STATUS_INVALID_HANDLE,
    PFAC_STATUS_INVALID_PARAMETER, 
    PFAC_STATUS_PATTERNS_NOT_READY,
    PFAC_STATUS_FILE_OPEN_ERROR,
    PFAC_STATUS_LIB_NOT_EXIST,   
    PFAC_STATUS_ARCH_MISMATCH,
    PFAC_STATUS_MUTEX_ERROR,
    PFAC_STATUS_INTERNAL_ERROR 
} PFAC_status_t ;

typedef enum {
    PFAC_PLATFORM_GPU = 0,  // default
    PFAC_PLATFORM_CPU = 1,
    PFAC_PLATFORM_CPU_OMP = 2
} PFAC_platform_t ;

typedef enum {
    PFAC_AUTOMATIC   = 0,  // default
    PFAC_TEXTURE_ON  = 1,
    PFAC_TEXTURE_OFF = 2
} PFAC_textureMode_t ;

typedef struct _spfac_userdata{
    unsigned ref_count;
    void *id;

} PFAC_USERDATA;

typedef struct _pfac_pattern{
    struct _pfac_pattern *next;
    unsigned char *patrn;
    unsigned char *casepatrn;
    int n;
    int nocase;
    int offset;
    int depth;
    int negative;
    PFAC_USERDATA *udata;
    int iid;
    void *rule_option_tree;
    void *neg_list;
} PFAC_PATTERN;

struct PFAC_STRUCT;

typedef struct PFAC_STRUCT* PFAC_handle_t;

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
PFAC_status_t  PFAC_create( PFAC_handle_t handle );
void  PFAC_freeTable( PFAC_handle_t handle );
void  PFAC_freeResource( PFAC_handle_t handle );
PFAC_status_t  PFAC_destroy( PFAC_handle_t handle );

/*
 *  suppose transistion table has S states, labelled as s0, s1, ... s{S-1}
 *  and Bj denotes number of valid transition of s{i}
 *  for each state, we use sj >= Bj^2 locations to contain Bj transistion.
 *  In order to avoid collision, we choose a value k and a prime p such that
 *  (k*x mod p mod sj) != (k*y mod p mod sj) for all characters x, y such that 
 *  (s{j}, x) and (s{j}, y) are valid transitions.
 *  
 *  Hash table consists of rowPtr and valPtr, similar to CSR format.
 *  valPtr contains all transitions and rowPtr[i] contains offset pointing to valPtr.
 *
 *  Element of rowPtr is int2, which is equivalent to
 *  typedef struct{
 *     int offset ;
 *     int k_sminus1 ;
 *  } 
 *
 *  sj is power of 2 and less than 256, and 0 < kj < 256, so we can encode (k,s-1) by a
 *  32-bit integer, k occupies most significant 16 bits and (s-1) occupies Least significant 
 *  16 bits.
 *
 *  sj is power of 2 and we need to do modulo s, in order to speedup, we use mask to do 
 *  modulo, say x mod s = x & (s-1)
 *
 *  Element of valPtr is int2, equivalent to
 *  tyepdef struct{
 *     int nextState ;
 *     int ch ;
 *  } 
 *
 *
 */
    
typedef struct {
    int nextState;
    int ch;
} TableEle;

#define  CHAR_SET    256
#define  TRAP_STATE  0xFFFFFFFF

#define  FILENAME_LEN    256

PFAC_status_t PFAC_tex_mutex_lock(PFAC_handle_t handle);

PFAC_status_t PFAC_tex_mutex_unlock(PFAC_handle_t handle);

typedef PFAC_status_t (*PFAC_kernel_protoType)( 
    PFAC_handle_t handle, 
    char *d_input_string, 
    size_t input_size,
    int *d_matched_result,
    int *d_num_matched ) ;

struct PFAC_STRUCT {
    // host
    char **rowPtr ; /* rowPtr[0:k-1] contains k pointer pointing to k patterns which reside in "valPtr"
                     * the order of patterns is sorted by lexicographic, say
                     *     rowPtr[i] < rowPtr[j]
                     *  if either rowPtr[i] = prefix of rowPtr[j] but length(rowPtr[i]) < length(rowPtr[j])
                     *     or \alpha = prefix(rowPtr[i])=prefix(rowPtr[j]) such that
                     *        rowPtr[i] = [\alpha]x[beta]
                     *        rowPtr[j] = [\aloha]y[gamma]
                     *     and x < y
                     *
                     *  pattern ID starts from 1 and follows the order of patterns in input file.
                     *  We record pattern ID in "patternID_table" and legnth of pattern in "patternLen_table".
                     *
                     *  for example, pattern rowPtr[0] = ABC, it has length 3 and ID = 5, then
                     *  patternID_table[0] = 5, and patternLen_table[5] = 3
                     *
                     *  WARNING: pattern ID starts by 1, so patternLen_table[0] is useless, in order to pass
                     *  valgrind, we reset patternLen_table[0] = 0
                     *
                     */
    char *valPtr ;  // contains all patterns, each pattern is terminated by null character '\0'
    int *patternLen_table ;
    int *patternID_table ;

    vector< vector<TableEle> > *table_compact;
    
    int  *h_PFAC_table ; /* explicit 2-D table */

    // int2 *h_hashRowPtr ;
    // int2 *h_hashValPtr ;
    int  *h_tableOfInitialState ;
    int  hash_p ; // p = 2^m + 1 
    int  hash_m ;

    // device
    int  *d_PFAC_table ; /* explicit 2-D table */

    // int2 *d_hashRowPtr ;
    // int2 *d_hashValPtr ;
    int  *d_tableOfInitialState ; /* 256 transition function of initial state */

    size_t  numOfTableEntry ; 
    size_t  sizeOfTableEntry ; 
    size_t  sizeOfTableInBytes ; // numOfTableEntry * sizeOfTableEntry
       
    // function pointer of non-reduce kernel
    PFAC_kernel_protoType  kernel_ptr;

    // function pointer of reduce kernel under PFAC_TIME_DRIVEN
    // PFAC_reduce_kernel_protoType  reduce_kernel_ptr ;
    
    // function pointer of reduce kernel under PFAC_SPACE_DRIVEN
    // PFAC_reduce_kernel_protoType  reduce_inplace_kernel_ptr ;

    int maxPatternLen ; /* maximum length of all patterns
                         * this number can determine which kernel is proper,
                         * for example, if maximum length is smaller than 512, then
                         * we can call a kernel with smem
                         */
                             
    int  max_numOfStates = 00 ; // maximum number of states, this is an estimated number from size of pattern file
    int  numOfPatterns ;  // number of patterns
    int  numOfStates ; // total number of states in the DFA, states are labelled as s0, s1, ..., s{state_num-1}
    int  numOfFinalStates ; // number of final states
    int  initial_state ; // state id of initial state

    int  numOfLeaves ; // number of leaf nodes of transistion table. i.e nodes without fan-out
                       // numOfLeaves <= numOfFinalStates

    /* warpper for pthread_mutex_lock and pthread_mutex_unlock */
    mutex  __pfac_tex_mutex;
    
    int  platform ;
    
    int  perfMode ;
    
    int  textureMode ;
    
    bool isPatternsReady ;
    
    int device_no ; // = 10*deviceProp.major + deviceProp.minor ;
    
    char patternFile[FILENAME_LEN] ;

    PFAC_PATTERN *pfacPatterns;

    void (*userfree)(void *p);
    void (*optiontreefree)(void **p);
    void (*neg_list_free)(void **p);
}  ;

struct patternEle{
    char *patternString;
    int patternID;
};



/*
 *  return
 *  ------
 *  char * pointer to a NULL-terminated string. This is string literal, do not overwrite it.
 *
 */
static const char* PFAC_getErrorString( PFAC_status_t status )
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
    // if ( PFAC_STATUS_BASE > status ){
    //     return cudaGetErrorString( (cudaError_t) status ) ;
    // }

    switch(status){
        case PFAC_STATUS_ALLOC_FAILED:
            return PFAC_alloc_failed_str ;
        case PFAC_STATUS_CUDA_ALLOC_FAILED:
            return PFAC_cuda_alloc_failed_str;
        case PFAC_STATUS_INVALID_HANDLE:
            return PFAC_invalid_handle_str ;
        case PFAC_STATUS_INVALID_PARAMETER:
            return PFAC_invalid_parameter_str ;
        case PFAC_STATUS_PATTERNS_NOT_READY:
            return PFAC_patterns_not_ready_str ;
        case PFAC_STATUS_FILE_OPEN_ERROR:
            return PFAC_file_open_error_str ;
        case PFAC_STATUS_LIB_NOT_EXIST:
            return PFAC_lib_not_exist_str ;
        case PFAC_STATUS_ARCH_MISMATCH:
            return PFAC_arch_mismatch_str ;
        case PFAC_STATUS_MUTEX_ERROR:
            return PFAC_mutex_error ;
        default : // PFAC_STATUS_INTERNAL_ERROR:
            return PFAC_internal_error_str ;
    }
}

/**
 * 
 * SNORT PROTOTYPE
 * 
 */

PFAC_STRUCT * pfacNew (void (*userfree)(void *p),
        void (*optiontreefree)(void **p),
        void (*neg_list_free)(void **p));

int pfacAddPattern( PFAC_STRUCT * p, unsigned char * pat, int n,
        int nocase, int offset, int depth, int negative, void * id, int iid );

int pfacCompile ( PFAC_STRUCT * pfac,
        int (*build_tree)(void * id, void **existing_tree),
        int (*neg_list_func)(void *id, void **list));

int pfacSearch ( PFAC_STRUCT * pfac,unsigned char * T, int n, 
        int (*Match)(void * id, void *tree, int index, void *data, void *neg_list),
        void * data, int* current_state );

void pfacFree ( PFAC_STRUCT * pfac );
int pfacPatternCount ( PFAC_STRUCT * pfac );

int pfacPrintDetailInfo(PFAC_STRUCT *);

int pfacPrintSummaryInfo(void);

// #ifdef __cplusplus
// }
// #endif   // __cplusplus


#endif   // PFAC_H_
