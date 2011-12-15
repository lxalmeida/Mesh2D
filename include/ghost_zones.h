#ifndef GHOST_ZONE_H
#define GHOST_ZONE_H

#include <iostream>
#include <halo.h>
#include <topology.h>
#include <resource.h>
#include <fstream>
#include <sstream>

using namespace std;

class Ghost_Zones {
    private:
        int rows, cols, num_halos;
        struct mesh_t mesh;
        Topology &topology;
        bool do_sync;
        void update_diag(void);
        void sync_update_diag(void);
        // Buffers para armazenar os dados dos vizinhos das diagonais
        struct diag_buff_t diag_top_left, diag_top_right,
                           diag_bottom_left, diag_bottom_right;
        void alloc_diags(void);
        void print_buff(struct diag_buff_t *buff);
    public:
        Ghost_Zones(int m, int n, struct mesh_t mesh, Topology &topology, int num_halos);
        ~Ghost_Zones();
        void init_halos(void);
        bool update(void);
        void sync(void);
        void print_halo(void);
        int get_num_halos(void);
        Halo_Left *left;
        Halo_Right *right;
        Halo_Bottom *bottom;
        Halo_Top *top;
};

#endif
