#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iterator>

using namespace std;

#define PLAIN_TYPE_STENCIL 0
#define STAR_TYPE_STENCIL 1

#define LOCATION_TOP 1
#define LOCATION_BOTTOM 2
#define LOCATION_MIDDLE (LOCATION_TOP | LOCATION_BOTTOM)

#define LEFT   0
#define RIGHT  1
#define TOP    2
#define BOTTOM 3
#define FRONT  4
#define BACK   5
#define TOP_LEFT     6
#define TOP_RIGHT    7
#define BOTTOM_LEFT  8
#define BOTTOM_RIGHT 9

#define LEFT_HALO_TAG   LEFT
#define RIGHT_HALO_TAG  RIGHT
#define TOP_HALO_TAG    TOP
#define BOTTOM_HALO_TAG BOTTOM
#define FRONT_HALO_TAG  FRONT
#define BACK_HALO_TAG   BACK
#define TOP_LEFT_CORNER_TAG TOP_LEFT
#define TOP_RIGHT_CORNER_TAG TOP_RIGHT
#define BOTTOM_LEFT_CORNER_TAG BOTTOM_LEFT
#define BOTTOM_RIGHT_CORNER_TAG BOTTOM_RIGHT

#define TO_LEFT_NEIGHBOR_HALO_TAG   LEFT_HALO_TAG
#define TO_RIGHT_NEIGHBOR_HALO_TAG  RIGHT_HALO_TAG
#define TO_TOP_NEIGHBOR_HALO_TAG    TOP_HALO_TAG
#define TO_BOTTOM_NEIGHBOR_HALO_TAG BOTTOM_HALO_TAG
#define TO_FRONT_NEIGHBOR_HALO_TAG  FRONT_HALO_TAG
#define TO_BACK_NEIGHBOR_HALO_TAG   BACK_HALO_TAG
#define TO_TOP_LEFT_NGBR_TAG        BOTTOM_RIGHT_CORNER_TAG
#define TO_TOP_RIGHT_NGBR_TAG       BOTTOM_LEFT_CORNER_TAG
#define TO_BOTTOM_LEFT_NGBR_TAG     TOP_RIGHT_CORNER_TAG
#define TO_BOTTOM_RIGHT_NGBR_TAG    TOP_LEFT_CORNER_TAG

// Para armazenar as coordenadas do processo na topologia
struct coords {
    int x, y, z;
};
// Para armazenar o número de processos em cada dimensão
struct dims {
    int x, y, z;
};
// Para armazenar o rank dos vizinhos em todas as direções
struct neighbors {
    int x_right, x_left;
    int y_top, y_bottom;
    int z_front, z_back;
};

struct neighbor {
    int direction;   // Direção do vizinho: left, right, top, bottom
    int rank;        // Rank do vizinho
    MPI_Request req; // Requisição para as primitivas MPI_Isend/Irecv 
};

/**
 * \brief Iterador para os vizinhos.
 * 
 */
class neighbors_iterator : public iterator<forward_iterator_tag, struct neighbor> {
    private:
        struct neighbor *n;
    public:
        neighbors_iterator(struct neighbor *p) : n(p) { };
        neighbors_iterator& operator++() {++n; return *this;}
        neighbors_iterator& operator++(int) {n++; return *this;}
        bool operator==(const neighbors_iterator& rhs) { return n->rank == (rhs.n)->rank; }
        bool operator!=(const neighbors_iterator& rhs) { return n->rank != (rhs.n)->rank; }
        struct neighbor& operator*() { return *n; }
};

/**
 * \brief Classe que abstrai uma topologia cartesiana MPI bidimensional.
 * 
 */
class Topology {
    private:
        struct coords coords; // Coordenadas do processo na topologia
        int *periods, ndims;  // Numero de dimensoes
        struct dims dims;     // Número de processos em cada dimensão
        MPI_Comm comm;        // Comunicador da topologia
        int rank;             // Rank do processo na topologia
        int nprocs;           // Numero de processos na topologia
        bool has_diag;
        struct neighbor *neighbors; // Rank dos vizinhos do processo
        int num_neighbors;
        int init(MPI_Comm comm_old, int stencil_type, int ndims, int *periods, int reorder);
        int meet_the_neighbors(void);
        void set_coords(int coords[]);
        void set_ndims(int ndims);
        void set_dims(int dims[]);
    public:
        Topology(int ndims, int stencil_type = PLAIN_TYPE_STENCIL, int periods[] = NULL, int reorder = 0, MPI_Comm comm_old = MPI_COMM_WORLD);
        ~Topology(); // Destrutor
        void create_diag_neighbors(void);
        MPI_Comm get_comm(void); // Retorna o comunicador da topologia
        int get_rank(void); // Retorna o rank do processo
        int get_nprocs(void); // Retorna o número de processos que compõe a topologia
        int get_ndims(void); // Retorna o número de dimensoes da topologia
        struct dims get_dims(void); // Retorna o número de processos em cada dimensão
        struct coords get_coords(void); // Retorna as coordenadas do processo na topologia
        bool has_diagonal(void);
        neighbors_iterator first_neighbor(void); // Retorna um iterator para os vizinhos
        neighbors_iterator last_neighbor(void); // Retorna o último vizinho
};

#endif

