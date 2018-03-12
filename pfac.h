#include <stdio.h>

#include <vector>

#ifndef PFAC_H_
#define PFAC_H_

#ifdef __cplusplus
extern "C" {
#endif   // __cplusplus

/* This is missing from very old Linux libc. */
#ifndef RTLD_NOW
#define RTLD_NOW 2
#endif

#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 255
#endif

using namespace std;

/*
 * debug mode:  PFAC_PRINTF( ... ) printf( __VA_ARGS__ )
 * release mode:  PFAC_PRINTF( ... ) 
 */
#ifndef DEBUG 
    #define PFAC_PRINTF(...)
#else
    #define PFAC_PRINTF( ... ) printf( __VA_ARGS__ )
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

struct PFAC_context ;

typedef struct PFAC_context* PFAC_handle_t ;

PFAC_status_t  PFAC_CPU_OMP(PFAC_handle_t handle, char *input_string, const int input_size, int *h_matched_result );
PFAC_status_t  PFAC_CPU( PFAC_handle_t handle, char *h_input_string, const int input_size, int *h_matched_result ) ;

void  PFAC_freeTable( PFAC_handle_t handle );
void  PFAC_freeResource( PFAC_handle_t handle );
PFAC_status_t  PFAC_bindTable( PFAC_handle_t handle );
PFAC_status_t  PFAC_create2DTable( PFAC_handle_t handle );
PFAC_status_t  PFAC_createHashTable( PFAC_handle_t handle );

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
    int nextState ; 
    int ch ;
} TableEle ; 

#define  CHAR_SET    256
#define  TRAP_STATE  0xFFFFFFFF

#define  FILENAME_LEN    256

typedef PFAC_status_t (*PFAC_kernel_protoType)( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result ) ;

struct PFAC_context {
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

    int2 *h_hashRowPtr ;
    int2 *h_hashValPtr ;
    int  *h_tableOfInitialState ;
    int  hash_p ; // p = 2^m + 1 
    int  hash_m ;

    // device
    int  *d_PFAC_table ; /* explicit 2-D table */

    int2 *d_hashRowPtr ;
    int2 *d_hashValPtr ;
    int  *d_tableOfInitialState ; /* 256 transition function of initial state */

    size_t  numOfTableEntry ; 
    size_t  sizeOfTableEntry ; 
    size_t  sizeOfTableInBytes ; // numOfTableEntry * sizeOfTableEntry
       
    // function pointer of non-reduce kernel under PFAC_TIME_DRIVEN
    PFAC_kernel_protoType  kernel_time_driven_ptr ;
    
    // function pointer of non-reduce kernel under PFAC_SPACE_DRIVEN
    // PFAC_kernel_protoType  kernel_space_driven_ptr ;
    
    // function pointer of reduce kernel under PFAC_TIME_DRIVEN
    // PFAC_reduce_kernel_protoType  reduce_kernel_ptr ;
    
    // function pointer of reduce kernel under PFAC_SPACE_DRIVEN
    // PFAC_reduce_kernel_protoType  reduce_inplace_kernel_ptr ;

    int maxPatternLen ; /* maximum length of all patterns
                         * this number can determine which kernel is proper,
                         * for example, if maximum length is smaller than 512, then
                         * we can call a kernel with smem
                         */
                             
    int  max_numOfStates ; // maximum number of states, this is an estimated number from size of pattern file
    int  numOfPatterns ;  // number of patterns
    int  numOfStates ; // total number of states in the DFA, states are labelled as s0, s1, ..., s{state_num-1}
    int  numOfFinalStates ; // number of final states
    int  initial_state ; // state id of initial state

    int  numOfLeaves ; // number of leaf nodes of transistion table. i.e nodes without fan-out
                       // numOfLeaves <= numOfFinalStates
    
    int  platform ;
    
    int  perfMode ;
    
    int  textureMode ;
    
    bool isPatternsReady ;
    
    int device_no ; // = 10*deviceProp.major + deviceProp.minor ;
    
    char patternFile[FILENAME_LEN] ;
}  ;

/*
 *  return
 *  ------
 *  char * pointer to a NULL-terminated string. This is string literal, do not overwrite it.
 *
 */
const char* PFAC_getErrorString( PFAC_status_t status ) ;

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
PFAC_status_t  PFAC_readPatternFromFile( PFAC_handle_t handle, char *filename ) ;

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


PFAC_status_t PFAC_tex_mutex_lock( void );

PFAC_status_t PFAC_tex_mutex_unlock( void );

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS            if operation is successful
 *  PFAC_STATUS_INTERNAL_ERROR     please report bugs
 *
 */
PFAC_status_t  PFAC_dumpTransitionTable( PFAC_handle_t handle, FILE *fp ) ;

/* KERNEL */

extern PFAC_status_t  PFAC_kernel_timeDriven_wrapper( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result) ;





#ifdef __cplusplus
}
#endif   // __cplusplus


#endif   // PFAC_H_