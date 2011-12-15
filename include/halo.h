#ifndef HALO_H
#define HALO_H

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <resource.h>
#include <string.h>
#include <pthread.h>
#include <iterator>
#include <sstream>

// Posições do Halo com relação ao seu subdomínio
#define LEFT   0
#define RIGHT  1
#define TOP    2
#define BOTTOM 3

#define HALO_THICKNESS 1

//#define DEBUG_COMM_CORNERS

using namespace std;

struct mesh_t {
    double ***mesh;
    int size_x, size_y;
};

struct diag_buff_t {
    double **buff;
    int nrows, start_row, end_row;
    int ncols, start_col, end_col;
    int rank_source;
};

/**
 * \brief Iterador para os Halos.
 * 
 */
class halo_iterator : public iterator<forward_iterator_tag, struct neighbor> {
private:
    double *curr;
    int stride;
public:
    halo_iterator(double *curr, int stride) : curr(curr), stride(stride) {
    }
    // Operador de pré-incremento
    halo_iterator & operator++() {
        ++curr;
        return *this;
    }
    // Operador de pós-incremento
    halo_iterator & operator++(int) {
        curr++;
        return *this;
    }
    // Operador de comparação ==
    bool operator==(const halo_iterator& rhs) {
        return curr == rhs.curr;
    }
    // Operador de comparação !=
    bool operator!=(const halo_iterator& rhs) {
        return curr != rhs.curr;
    }
    // Operador de desreferenciação
    double& operator*() {
        return *curr;
    }
    ///
    // Redução dos valores do stencil
    // Este método depende de como o estêncil deve ser calculado
    double reduce(void) {
        return *(curr - 1) + *(curr + 1) + *(curr - stride) + *(curr + stride);
    }
};

class H {
private:
    double ***halo;
    const int halo_id, start, size, stride;
public:

    H(double ***halo, int halo_id, int start, int size, int stride) : halo(halo),
    halo_id(halo_id),
    start(start),
    size(size),
    stride(stride) {
    }

    ~H() {
    }

    halo_iterator first_inner(void) {
        return halo_iterator((double*) ((*halo)[halo_id] + start + HALO_THICKNESS), this->stride);
    }

    halo_iterator last_inner(void) {
        return halo_iterator((double*) ((*halo)[halo_id] + (size - 1) - HALO_THICKNESS), this->stride);
    }

    halo_iterator first(void) {
        return halo_iterator((double*) ((*halo)[halo_id] + start), this->stride);
    }

    halo_iterator last(void) {
        return halo_iterator((double*) ((*halo)[halo_id] + (size - 1)), this->stride);
    }
};

/**
 * Classe Halo, que abstrai as regiões de sobreposição entre os subdomínios.
 *  
 */
class Halo_Left { // "Auréola; Anel"
private:
    double **data_u, **data_v; /**< Armazena as posições da Halo. */
    double **data_source, **data_dest;
    double **buff;
    struct diag_buff_t corner_top, corner_bottom;
    H **h_source, **h_dest;
    struct mesh_t mesh;
    int halo_position; /**< Define a posição do Halo com relação ao mesh. Pode ser LEFT, RIGHT, TOP ou BOTTOM. Essa
                                definição é utilizada para definir o modo de tratamento do array bidimensional que armazena
                                o Halo. Por exemplo, se o Halo for a região de sobreposição esquerda, então o array bidimensional
                                deverá ser tratado como halo_size x NGZ, onde NGZ é o número de regiões de sobreposição e halo_size
                                é o tamanho do Halo (isso é, num_linhas x num_colunas). */
    int start_row, end_row, start_col, end_col;
    int num_iterators_ctrl;
    int num_iterators;
    int num_rows, num_cols;
    int halo_size; /**< Tamanho da região de sobreposição */
    int num_halos; /**< Número de regiões de sobreposição */
    void alloc(void); /**< Aloca do vetor bidimensional que conterá os dados da Halo. */
    void set_iterators(void);
    void swap(void);
    void print_buff(double **, int m, int n);
    MPI_Comm comm; /**< Comunicador para o MPI_Irecv */
    MPI_Request request; // Request para Irecv
    int rank_neighbor, direction_tag; // Rank de quem esperar pela mensagem
    // e tag da mensagem
    ///////////////////
    friend void* jacobi2d_left(void *args);
    pthread_attr_t thread_attr;
    pthread_t thread_id;
    ///////////////////
public:
    // Um halo pode não ter vizinhos (não estar sobreposto)
    Halo_Left(int halo_size, int num_halos, struct mesh_t mesh,
            struct diag_buff_t *corner_top, struct diag_buff_t *corner_bottom,
            int rank_neighbor = MPI_PROC_NULL, int direction_tag = -1,
            MPI_Comm comm = MPI_COMM_WORLD);
    ~Halo_Left();
    void set_neighbor_info(int rank_neighbor, int direction_tag, MPI_Comm comm); // Informa os dados do vizinho
    int get_neighbor_rank(void);
    int get_halo_size(void);
    int get_num_halos(void);
    bool transfer(void);
    bool init_update(void); // Inicializa o Irecv
    void sync(void); // Espera pelo Irecv
    double& operator[](int i);
    double operator[](int i) const;
    void print_to_file(const char *filename);
    void print_iter_to_file(const char *filename);
    fstream logfile;
    void print_info(void);

};

/**
 * Classe Halo, que abstrai as regiões de sobreposição entre os subdomínios.
 *  
 */
class Halo_Right { // "Auréola; Anel"
private:
    double **data_u, **data_v; /**< Armazena as posições da Halo. */
    double **data_source, **data_dest;
    double **buff;
    struct diag_buff_t corner_top, corner_bottom;
    H **h_source, **h_dest;
    struct mesh_t mesh;
    int halo_position; /**< Define a posição do Halo com relação ao mesh. Pode ser LEFT, RIGHT, TOP ou BOTTOM. Essa
                        definição é utilizada para definir o modo de tratamento do array bidimensional que armazena
                        o Halo. Por exemplo, se o Halo for a região de sobreposição esquerda, então o array bidimensional
                        deverá ser tratado como halo_size x NGZ, onde NGZ é o número de regiões de sobreposição e halo_size
                        é o tamanho do Halo (isso é, num_linhas x num_colunas). */
    int start_row, end_row, start_col, end_col;
    int num_iterators_ctrl;
    int num_iterators;
    int num_rows, num_cols;
    int halo_size; /**< Tamanho da região de sobreposição */
    int num_halos; /**< Número de regiões de sobreposição */
    void alloc(void); /**< Aloca do vetor bidimensional que conterá os dados da Halo. */
    void set_iterators(void);
    void swap(void);
    void print_buff(double **, int m, int n);
    MPI_Comm comm; /**< Comunicador para o MPI_Irecv */
    MPI_Request request; // Request para Irecv
    int rank_neighbor, direction_tag; // Rank de quem esperar pela mensagem
    // e tag da mensagem
    ///////////////////
    friend void* jacobi2d_right(void *args);
    pthread_attr_t thread_attr;
    pthread_t thread_id;
    ///////////////////
public:
    // Um halo pode não ter vizinhos (não estar sobreposto)
    Halo_Right(int halo_size, int num_halos, struct mesh_t mesh,
            struct diag_buff_t *corner_top, struct diag_buff_t *corner_bottom,
            int rank_neighbor = MPI_PROC_NULL, int direction_tag = -1,
            MPI_Comm comm = MPI_COMM_WORLD);
    ~Halo_Right();
    void set_neighbor_info(int rank_neighbor, int direction_tag, MPI_Comm comm); // Informa os dados do vizinho
    int get_neighbor_rank(void);
    int get_halo_size(void);
    int get_num_halos(void);
    bool transfer(void);
    bool init_update(void); // Inicializa o Irecv
    void sync(void); // Espera pelo Irecv
    double& operator[](int i);
    double operator[](int i) const;
    void print_to_file(const char *filename);
    void print_iter_to_file(const char *filename);
    fstream logfile;
    void print_info(void);
};

/**
 * Classe Halo, que abstrai as regiões de sobreposição entre os subdomínios.
 *  Halo_Top
 */
class Halo_Top { // "Auréola; Anel"
private:
    double **data_u, **data_v; /**< Armazena as posições da Halo. */
    double **data_source, **data_dest;
    double **buff;
    struct diag_buff_t corner_left, corner_right;
    H **h_source, **h_dest;
    struct mesh_t mesh;
    int halo_position; /**< Define a posição do Halo com relação ao mesh. Pode ser LEFT, RIGHT, TOP ou BOTTOM. Essa
                        definição é utilizada para definir o modo de tratamento do array bidimensional que armazena
                        o Halo. Por exemplo, se o Halo for a região de sobreposição esquerda, então o array bidimensional
                        deverá ser tratado como halo_size x NGZ, onde NGZ é o número de regiões de sobreposição e halo_size
                        é o tamanho do Halo (isso é, num_linhas x num_colunas). */
    int start_row, end_row, start_col, end_col;
    int num_iterators_ctrl;
    int num_iterators;
    int num_rows, num_cols;
    int halo_size; /**< Tamanho da região de sobreposição */
    int num_halos; /**< Número de regiões de sobreposição */
    void alloc(void); /**< Aloca do vetor bidimensional que conterá os dados da Halo. */
    void set_iterators(void);
    void swap(void);
    void print_buff(double **, int m, int n);
    MPI_Comm comm; /**< Comunicador para o MPI_Irecv */
    MPI_Request request; // Request para Irecv
    int rank_neighbor, direction_tag; // Rank de quem esperar pela mensagem
    // e tag da mensagem
    ///////////////////
    friend void* jacobi2d_top(void *args);
    pthread_attr_t thread_attr;
    pthread_t thread_id;
    ///////////////////
public:
    // Um halo pode não ter vizinhos (não estar sobreposto)
    Halo_Top(int halo_size, int num_halos, struct mesh_t mesh,
            struct diag_buff_t *corner_left, struct diag_buff_t *corner_right,
            int rank_neighbor = MPI_PROC_NULL, int direction_tag = -1,
            MPI_Comm comm = MPI_COMM_WORLD);
    ~Halo_Top();
    void set_neighbor_info(int rank_neighbor, int direction_tag, MPI_Comm comm); // Informa os dados do vizinho
    int get_neighbor_rank(void);
    int get_halo_size(void);
    int get_num_halos(void);
    bool transfer(void);
    bool init_update(void); // Inicializa o Irecv
    void sync(void); // Espera pelo Irecv
    double& operator[](int i);
    double operator[](int i) const;
    void print_to_file(const char *filename);
    void print_iter_to_file(const char *filename);
    fstream logfile;
    void print_info(void);
};

/**
 * Classe Halo, que abstrai as regiões de sobreposição entre os subdomínios.
 *  Halo_Bottom
 */
class Halo_Bottom { // "Auréola; Anel"
private:
    double **data_u, **data_v; /**< Armazena as posições da Halo. */
    double **data_source, **data_dest;
    double **buff;
    struct diag_buff_t corner_left, corner_right;
    H **h_source, **h_dest;
    struct mesh_t mesh;
    int halo_position; /**< Define a posição do Halo com relação ao mesh. Pode ser LEFT, RIGHT, TOP ou BOTTOM. Essa
                        definição é utilizada para definir o modo de tratamento do array bidimensional que armazena
                        o Halo. Por exemplo, se o Halo for a região de sobreposição esquerda, então o array bidimensional
                        deverá ser tratado como halo_size x NGZ, onde NGZ é o número de regiões de sobreposição e halo_size
                        é o tamanho do Halo (isso é, num_linhas x num_colunas). */
    int start_row, end_row, start_col, end_col;
    int num_iterators_ctrl;
    int num_iterators;
    int num_rows, num_cols;
    int halo_size; /**< Tamanho da região de sobreposição */
    int num_halos; /**< Número de regiões de sobreposição */
    void alloc(void); /**< Aloca do vetor bidimensional que conterá os dados da Halo. */
    void set_iterators(void);
    void swap(void);
    void print_buff(double **, int m, int n);
    MPI_Comm comm; /**< Comunicador para o MPI_Irecv */
    MPI_Request request; // Request para Irecv
    int rank_neighbor, direction_tag; // Rank de quem esperar pela mensagem
    // e tag da mensagem
    ///////////////////
    friend void* jacobi2d_bottom(void *args);
    pthread_attr_t thread_attr;
    pthread_t thread_id;
    ///////////////////
public:
    // Um halo pode não ter vizinhos (não estar sobreposto)
    Halo_Bottom(int halo_size, int num_halos, struct mesh_t mesh,
            struct diag_buff_t *corner_left, struct diag_buff_t *corner_right,
            int rank_neighbor = MPI_PROC_NULL, int direction_tag = -1,
            MPI_Comm comm = MPI_COMM_WORLD);
    ~Halo_Bottom();
    void set_neighbor_info(int rank_neighbor, int direction_tag, MPI_Comm comm); // Informa os dados do vizinho
    int get_neighbor_rank(void);
    int get_halo_size(void);
    int get_num_halos(void);
    bool transfer(void);
    bool init_update(void); // Inicializa o Irecv
    void sync(void); // Espera pelo Irecv
    double& operator[](int i);
    double operator[](int i) const;
    void print_to_file(const char *filename);
    void print_iter_to_file(const char *filename);
    fstream logfile;
    void print_info(void);
};

#endif

