#include "pfac.h"
#include "pfac_file.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <cstring>
#include "pfac_table.h"

#include <dirent.h>

int MatchFound (void * id, void *tree, int index, void *data, void *neg_list)
{
    fprintf (stdout, "%s\n", (char *) id);
    return 0;
}

PFAC_status_t matchFromDir(PFAC_STRUCT *pfac, char* dirpath)
{
    PFAC_handle_t handle = (PFAC_handle_t) pfac;
    std::string strdir = std::string(dirpath);

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (dirpath)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            printf("File: %s\n", ent->d_name);
            matchFromFile(handle, (strdir + ent->d_name).c_str());
        }        
        closedir (dir);
    } else {
        printf ("Could not open directory: %s\n", dirpath);
        return PFAC_STATUS_FILE_OPEN_ERROR;
    }

    return PFAC_STATUS_SUCCESS;
}

int main(int argc, char **argv)
{
    char dumpTableFile[] = "table.txt";
    char inputDir[] = "/home/ical/projects/fyp/dump/payloads/";
    char patternFile[] = "/home/ical/projects/fyp/dump/rules/merged";
    PFAC_STRUCT *pfac;
    PFAC_status_t PFAC_status;
    int input_size;
    char *h_inputString = NULL;
    int  *h_matched_result = NULL;

    // create PFAC handle 
    pfac = pfacNew( NULL, NULL, NULL ) ;
    assert( pfac != NULL );

    // read patterns, compile patterns and dump transition table 
    PFAC_status = PFAC_readPatternFromFile( pfac, patternFile) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to read pattern from file, %s\n", PFAC_getErrorString(PFAC_status) );
        return -1;   
    }

    PFAC_status = matchFromDir(pfac, inputDir);
    if ( PFAC_STATUS_SUCCESS != PFAC_status ) {
        printf("Error: fails to match from dir, %s\n", PFAC_getErrorString(PFAC_status) );
        return -1;   
    }

    // dump transition table 
    // FILE *table_fp = fopen(dumpTableFile, "w");
    // assert(NULL != table_fp);
    // PFAC_status = PFAC_dumpTransitionTable( pfac, table_fp );
    // fclose(table_fp);
    // if ( PFAC_STATUS_SUCCESS != PFAC_status ) {
    //     printf("Error: fails to dump transition table, %s\n", PFAC_getErrorString(PFAC_status));
    //     return -1;
    // }

    // output matched result
    // for (int i = 0; i < input_size; i++) {
    //     if (h_matched_result[i] != 0) {
    //         printf("At position %4d, match pattern %d\n", i, h_matched_result[i]);
    //     }
    // }
    // printf("Pattern found: %d\n", count);

    pfacFree( pfac ) ;

    /*
     * Address consistency
     * https://stackoverflow.com/questions/6054271/freeing-pointers-from-inside-other-functions-in-c
     */
    pfac = NULL;

    return 0;
}
