#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cassert>

#include "pfac_table.h"
#include <cuda_runtime.h>

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

/* Custom comparator */
struct pattern_cmp_functor {

    // pattern *s and *t are terminated by character '\n'
    // strict weak ordering
    // return true if the first argument goes before the second argument
    bool operator()( patternEle pattern_s, patternEle pattern_t ){
        char s_char, t_char ;
        bool s_end, t_end ;
        int s_len = pattern_s.patternLen, t_len = pattern_t.patternLen;
        char *s_sweep = pattern_s.patternString;
        char *t_sweep = pattern_t.patternString;

        for (int i = 0; ; i++) {
            s_char = *s_sweep++ ;
            t_char = *t_sweep++ ;
            s_end = (i == s_len) ;
            t_end = (i == t_len) ;

            if ( s_end || t_end ) { break; }

            if (s_char < t_char) {
                return true;
            } else if ( s_char > t_char ) {
                return false;
            }
        }

        if (s_end == t_end) { // pattern s is the same as pattern t, the order is don't care
            return true;
        }
        else if (s_end) { // pattern s is prefix of pattern t
            return true;
        }
        else {
            return false; // pattern t is prefix of pattern s
        }
    }
}; 

/*
 *  fill pattern table,
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
PFAC_status_t PFAC_fillPatternTable ( PFAC_handle_t handle )
{
    if ( handle->valPtr == NULL ) {
        return PFAC_STATUS_INVALID_PARAMETER;
    }

    int numOfPatterns = handle->numOfPatterns;
    char *buffer = handle->valPtr;
    vector< struct patternEle > rowIdxArray;

    PFAC_PATTERN *plist;
    char *offset;
    for (plist = handle->pfacPatterns, offset = buffer;
        plist != NULL;
        offset += plist->n + 1, plist = plist->next)
    {
        memcpy(offset, plist->casepatrn, plist->n);
        rowIdxArray.push_back(patternEle(offset, 0, plist->n));
    }

    // sort patterns by lexicographic order
    std::sort(rowIdxArray.begin(), rowIdxArray.end(), pattern_cmp_functor());

    int rowIdxArraySize = rowIdxArray.size();
    handle->rowPtr = (char**)malloc(sizeof(char*) * rowIdxArraySize);
    handle->patternID_table = (int*)malloc(sizeof(int) * rowIdxArraySize);
    handle->patternLen_table = (int*)malloc(sizeof(int) * (rowIdxArraySize + 1));
    if ((NULL == handle->rowPtr) ||
        (NULL == handle->patternID_table) ||
        (NULL == handle->patternLen_table))
    {
        return PFAC_STATUS_ALLOC_FAILED;
    }

    // Compute f(final state) = patternID
    for (int i = 0; i < rowIdxArraySize; i++) {
        handle->rowPtr[i] = rowIdxArray[i].patternString;
        handle->patternID_table[i] = i + 1; // pattern number starts from 1

        // pattern ID starts from 1, so patternID = i+1
        handle->patternLen_table[i + 1] = rowIdxArray[i].patternLen;
    }

    return PFAC_STATUS_SUCCESS;
}

/*
 *  Prepare DFA transition table
 *  given sorted "patternListTable" 
 *
 */
PFAC_status_t PFAC_prepareTable ( PFAC_handle_t handle )
{
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
    PFAC_status_t status = create_PFACTable_spaceDriven((const char**)handle->rowPtr,
        (const int*)handle->patternLen_table, (const int*)handle->patternID_table,
        handle->max_numOfStates, handle->numOfPatterns, handle->initial_state, baseOfUsableStateID, 
        &handle->numOfStates, *(handle->table_compact) );

    if ( PFAC_STATUS_SUCCESS != status ){
        PFAC_freeResource( handle );
        return status ;
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

    status = PFAC_bindTable( handle ) ;
    if ( PFAC_STATUS_SUCCESS != status) {
        PFAC_freeResource( handle );
        handle->isPatternsReady = false ;
        return status ;
    }
        
    return status ;
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
