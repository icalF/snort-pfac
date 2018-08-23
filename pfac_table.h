#ifndef PFAC_TABLE_H_
#define PFAC_TABLE_H_

#include <ctype.h>

#include "pfac.h"

#ifdef __cplusplus
extern "C" {
#endif   // __cplusplus


/*
 ** Case Translation Table
 */
static unsigned char xlatcase[256];

/*
 *
 */
static void init_xlatcase ()
{
    int i;
    for (i = 0; i < 256; i++)
    {
        xlatcase[i] = (uint8_t)toupper (i);
    }
}

PFAC_status_t PFAC_fillPatternTable( PFAC_handle_t pfac );
PFAC_status_t PFAC_prepareTable( PFAC_handle_t pfac );

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
    vector< vector<TableEle> > &PFAC_table );

PFAC_status_t  PFAC_bindTable( PFAC_handle_t handle );
PFAC_status_t  PFAC_create2DTable( PFAC_handle_t handle );

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
PFAC_status_t  PFAC_bindTable( PFAC_handle_t handle );

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS            if operation is successful
 *  PFAC_STATUS_INTERNAL_ERROR     please report bugs
 *
 */
PFAC_status_t  PFAC_dumpTransitionTable( PFAC_handle_t handle, FILE *fp ) ;

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

#ifdef __cplusplus
}
#endif   // __cplusplus


#endif   // PFAC_TABLE_H_