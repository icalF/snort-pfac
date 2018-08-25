#ifndef PFAC_TEXTURE_H_
#define PFAC_TEXTURE_H_ 1

#include <texture_types.h>

extern textureReference *texRefTable;
extern texture < int, 1, cudaReadModeElementType > tex_PFAC_table;

#endif   // PFAC_TEXTURE_H_