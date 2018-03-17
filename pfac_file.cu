#include <algorithm>

#include "pfac_file.h"
#include "pfac_table.h"

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