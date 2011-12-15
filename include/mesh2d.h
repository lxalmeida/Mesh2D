#ifndef MESH2D_H
#define MESH2D_H

#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <topology.h>
#include <topology_output.h>
#include <ghost_zones.h>
#include <resource.h>
#include <fstream>
#include <sstream>
#include <pthread.h>

using namespace std;

#define PROC_RANGES_RESULT_TAG 255
#define PROC_RESULT_TAG 256

class Mesh;

struct args_t {
	Mesh *mesh;
	int lower_row;
	int upper_row;
	int lower_col;
	int upper_col;
	int border_lower_row;
	int border_upper_row;
	int border_lower_col;
	int border_upper_col;
	double val;
};

struct range_t {        // Limites e tamanhos de um certo mesh 
    int global_size_x, global_size_y; // Tamanho das dimensões x e y *de todo domínio*
    int size_x, size_y; // Tamanhos das dimensoes x e y *do subdominio*
    int start_x, end_x; // Limite inferior e superior da dimensao x 
    int start_y, end_y; // Limite inferior e superior da dimensao y 
};

class Mesh {
    private:
        double **data_u, **data_v;
        double **buff_left, **buff_right;
        Ghost_Zones *ghost_zones;
        struct range_t range;
        bool transfer;
        MPI_Request sreq_left, sreq_right, sreq_top, sreq_bottom;
        MPI_Request sreq_top_left, sreq_top_right, sreq_bottom_left, sreq_bottom_right;
        /* Tipo de dado para trocar o valor das bordas verticais */
        MPI_Datatype type_corner;
        void alloc_mesh(void);
        void free_mesh(void);
        void set_interval(int size, int rank, int nprocs, int *start, int *end);
        void copy_vals(double ***vals, int start_x, int end_x, int start_y, int end_y);
        void print_buff(fstream &dbg, double **buff, int start_row, int num_rows, int start_col, int num_cols);
        int num_threads;
        fstream logfile;
    public:
        Mesh(int m, int n);
        Mesh(int m, int n, int num_ghost_zones, int numthreads = 1);
        void init(int m, int n, int num_ghost_zones, int numthreads);
        ~Mesh();
        Topology *topology;
        double **data_source, **data_dest;
        double **final_result;
        void send_borders(void);
        void sync(void);
        double get_source_halo(int row, int col);
        int get_size_x(void);
        int get_size_y(void);
        int get_begin_x(void);
        int get_end_x(void);
        int get_begin_y(void);
        int get_end_y(void);
        void swap(void);
        void set_left_extern(double value);   // Left heat zone
        void set_right_extern(double value);  // Right heat zone
        void set_top_extern(double value);    // Top heat zone
        void set_bottom_extern(double value); // Bottom heat zone
        void gather(void);
        void print_final_result(void);
        void print_mesh_info(void);
        void print_file_mesh(void);
        void print_ghost_zones(void){ this->ghost_zones->print_halo(); };
        double& operator()(int row, int col);
        double operator()(int row, int col) const;
        void set_num_threads(int num_threads);
        int get_num_threads();
        // Isso deveria sair daqui
        pthread_attr_t thread_attr;
        pthread_t *thread_id;
        struct args_t *thread_args;
};

#endif

