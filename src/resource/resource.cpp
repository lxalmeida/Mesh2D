#include <resource.h>

/**  
 *  Função genérica que aloca um array bidimensional de forma contígua em
 *  memória.
 */
void* alloc_cont_array2d(double ***m, int rows, int cols){
    int i = 0;

    /* Aloca o vetor de ponteiros, que apontarão para o meio do array */
    (*m) = (double**)malloc(sizeof(double*) * rows);
    if((*m) == NULL){
        return(NULL);
    }
    /* Garante que as linhas da matriz sao contiguas em memoria */
    (*m)[0] = (double*)calloc(rows * cols, sizeof(double));
    if((*m)[0] == NULL){
        return(NULL);
    }
    /* Faz cada linha apontar para a posição correta no espaço contíguo */
    for(i = 1; i < rows; i++){
        (*m)[i] = (*m)[0] + (i * cols);
    }
    
    return((void*)(*m));
}

/** 
 *  Função genérica que libera um array bidimensional alocado através da função
 *  alloc_cont_array2d.
 */
void free_cont_array2d(double ***m){
    if(m == NULL || (*m) == NULL){
        return;
    }
    
    free((*m)[0]);
    free(*m);
    
    (*m) = NULL;
    
    return;
}

