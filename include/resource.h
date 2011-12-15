#ifndef PDE_RESOURCE_H
#define PDE_RESOURCE_H

#include <cstdlib>

void* alloc_cont_array2d(double ***m, int rows, int cols);
void free_cont_array2d(double ***m);

#endif

