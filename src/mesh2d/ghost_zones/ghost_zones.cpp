#include <ghost_zones.h>

/**
 * \brief Construtor das Ghost Zones.
 * 
 * Essa classe faz a amarração das Halos de um subdomínio, sendo responsável
 * pela instanciações desses objetos. As Halos podem ou não estarem sobrepostas
 * com vizinhos. Se não estiverem, as informações do vizinho não são informadas
 * à respectiva Halo. A não-sobreposição acontece quando um Halo é externa com
 * relação a todo domínio. Por exemplo, a borda mais a esquerda de todo domínio
 * não possui vizinho, mas mesmo assim há uma Halo ao lado dessa borda. Isso
 * acontece para todas as bordas.
 * 
 * @param [in] m  Número de linhas do subdomínio.
 * @param [in] n  Número de colunas do subdomínio.
 * @param [in] topology  Topologia da execução.
 * @param [in] num_halos  Número de regiões de sobreposição.
 * @return
 */
Ghost_Zones::Ghost_Zones(int m, int n, struct mesh_t mesh, Topology &topology,
        int num_halos) : mesh(mesh), topology(topology) {

    this->left = NULL;
    this->right = NULL;
    this->top = NULL;
    this->bottom = NULL;

    this->rows = m;
    this->cols = n;
    this->num_halos = num_halos;

    if (this->topology.has_diagonal()) {
        this->alloc_diags();
    }
    
    this->do_sync = false;
    this->init_halos();
}

void Ghost_Zones::alloc_diags(void) {
    int num_elements = this->num_halos * HALO_THICKNESS;

    neighbors_iterator neighbor = this->topology.first_neighbor();
    neighbors_iterator last_neighbor = this->topology.last_neighbor();

    this->diag_top_left.buff = NULL;
    this->diag_top_right.buff = NULL;
    this->diag_bottom_left.buff = NULL;
    this->diag_bottom_right.buff = NULL;

    this->diag_top_left.rank_source = MPI_PROC_NULL;
    this->diag_top_right.rank_source = MPI_PROC_NULL;
    this->diag_bottom_left.rank_source = MPI_PROC_NULL;
    this->diag_bottom_right.rank_source = MPI_PROC_NULL;

    while (neighbor != last_neighbor) {
        switch ((*neighbor).direction) {
            case TOP_LEFT:
                this->diag_top_left.nrows = num_elements;
                this->diag_top_left.ncols = num_elements;
                alloc_cont_array2d(&(this->diag_top_left.buff), this->diag_top_left.nrows, this->diag_top_left.ncols);
                this->diag_top_left.rank_source = (*neighbor).rank;
                break;

            case TOP_RIGHT:
                this->diag_top_right.nrows = num_elements;
                this->diag_top_right.ncols = num_elements;
                alloc_cont_array2d(&(this->diag_top_right.buff), this->diag_top_right.nrows, this->diag_top_right.ncols);
                this->diag_top_right.rank_source = (*neighbor).rank;
                break;

            case BOTTOM_LEFT:
                this->diag_bottom_left.nrows = num_elements;
                this->diag_bottom_left.ncols = num_elements;
                alloc_cont_array2d(&(this->diag_bottom_left.buff), this->diag_bottom_left.nrows, this->diag_bottom_left.ncols);
                this->diag_bottom_left.rank_source = (*neighbor).rank;
                break;

            case BOTTOM_RIGHT:
                this->diag_bottom_right.nrows = num_elements;
                this->diag_bottom_right.ncols = num_elements;
                alloc_cont_array2d(&(this->diag_bottom_right.buff), this->diag_bottom_right.nrows, this->diag_bottom_right.ncols);
                this->diag_bottom_right.rank_source = (*neighbor).rank;
                break;

            default:
                break;

        }
        neighbor++;
    }

    return;
}

Ghost_Zones::~Ghost_Zones() {
    if (this->left != NULL) {
        delete this->left;
    }

    if (this->right != NULL) {
        delete this->right;
    }

    if (this->top != NULL) {
        delete this->top;
    }

    if (this->bottom != NULL) {
        delete this->bottom;
    }
}

void Ghost_Zones::init_halos(void) {
    this->left = new Halo_Left(this->rows, num_halos, this->mesh, &(this->diag_top_left), &(this->diag_bottom_left));
    this->right = new Halo_Right(this->rows, num_halos, this->mesh, &(this->diag_top_right), &(this->diag_bottom_right));
    this->top = new Halo_Top(this->cols, num_halos, this->mesh, &(this->diag_top_left), &(this->diag_top_right));
    this->bottom = new Halo_Bottom(this->cols, num_halos, this->mesh, &(this->diag_bottom_left), &(this->diag_bottom_right));

    neighbors_iterator neighbor = this->topology.first_neighbor();
    neighbors_iterator last_neighbor = this->topology.last_neighbor();

    while (neighbor != last_neighbor) {
        switch ((*neighbor).direction) {
            case RIGHT:
                this->right->set_neighbor_info((*neighbor).rank, RIGHT_HALO_TAG, topology.get_comm());
                break;
            case LEFT:
                this->left->set_neighbor_info((*neighbor).rank, LEFT_HALO_TAG, topology.get_comm());
                break;
            case TOP:
                this->top->set_neighbor_info((*neighbor).rank, TOP_HALO_TAG, topology.get_comm());
                break;
            case BOTTOM:
                this->bottom->set_neighbor_info((*neighbor).rank, BOTTOM_HALO_TAG, topology.get_comm());
                break;
            default:
                break;
        }

        neighbor++;
    }

    return;
}

void Ghost_Zones::update_diag(void) {
    if (this->num_halos <= 1) {
        return;
    }

    neighbors_iterator neighbor = this->topology.first_neighbor();
    neighbors_iterator last_neighbor = this->topology.last_neighbor();

    while (neighbor != last_neighbor) {
        switch ((*neighbor).direction) {
            case TOP_LEFT:
                MPI_Irecv(this->diag_top_left.buff[0], // buff
                        this->diag_top_left.nrows * this->diag_top_left.ncols, // count
                        MPI_DOUBLE, // datatype
                        (*neighbor).rank, // source
                        TOP_LEFT_CORNER_TAG, // tag
                        topology.get_comm(), // comm
                        &((*neighbor).req)); // request
                break;
            case TOP_RIGHT:
                MPI_Irecv(this->diag_top_right.buff[0], // buff
                        this->diag_top_right.nrows * this->diag_top_right.ncols, // count
                        MPI_DOUBLE, // datatype
                        (*neighbor).rank, // source
                        TOP_RIGHT_CORNER_TAG, // tag
                        topology.get_comm(), // comm
                        &((*neighbor).req)); // request
                break;
            case BOTTOM_LEFT:
                MPI_Irecv(this->diag_bottom_left.buff[0], // buff
                        this->diag_bottom_left.nrows * this->diag_bottom_left.ncols, // count
                        MPI_DOUBLE, // datatype
                        (*neighbor).rank, // source
                        BOTTOM_LEFT_CORNER_TAG, // tag
                        topology.get_comm(), // comm
                        &((*neighbor).req)); // request
                break;
            case BOTTOM_RIGHT:
                MPI_Irecv(this->diag_bottom_right.buff[0], // buff
                        this->diag_bottom_right.nrows * this->diag_bottom_right.ncols, // count
                        MPI_DOUBLE, // datatype
                        (*neighbor).rank, // source
                        BOTTOM_RIGHT_CORNER_TAG, // tag
                        topology.get_comm(), // comm
                        &((*neighbor).req)); // request
                break;
            default:
                break;
        }

        neighbor++;
    }

    return;
}

void Ghost_Zones::sync_update_diag(void) {
    neighbors_iterator neighbor = this->topology.first_neighbor();
    neighbors_iterator last_neighbor = this->topology.last_neighbor();

    while (neighbor != last_neighbor) {
        switch ((*neighbor).direction) {
            case TOP_LEFT:
                if (MPI_Wait(&((*neighbor).req), MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                    cout << " MPI_Wait not successful" << endl;
                }
                break;
            case TOP_RIGHT:
                if (MPI_Wait(&((*neighbor).req), MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                    cout << " MPI_Wait not successful" << endl;
                }
                break;
            case BOTTOM_LEFT:
                if (MPI_Wait(&((*neighbor).req), MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                    cout << " MPI_Wait not successful" << endl;
                }
                break;
            case BOTTOM_RIGHT:
                if (MPI_Wait(&((*neighbor).req), MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                    cout << " MPI_Wait not successful" << endl;
                }
                break;
            default:
                break;
        }

        neighbor++;
    }
    
    return;
}

// Método invocado pelo Mesh2d para atualizar os halos, seja por recálculo,
// ou por transferência

bool Ghost_Zones::update(void) {
    this->do_sync = true;

    this->do_sync &= this->right->init_update();
    this->do_sync &= this->left->init_update();
    this->do_sync &= this->top->init_update();
    this->do_sync &= this->bottom->init_update();

    if (this->do_sync) {
        this->update_diag();
    }

    return this->do_sync;
}

void Ghost_Zones::sync(void) {
    if (this->do_sync) {
        this->sync_update_diag();
    }

    this->right->sync();
    this->left->sync();
    this->top->sync();
    this->bottom->sync();

    return;
}

int Ghost_Zones::get_num_halos(void) {
    return this->num_halos;
}

void Ghost_Zones::print_halo(void) {
    char file_halo_left[50], file_halo_right[50], file_halo_top[50], file_halo_bottom[50];

    sprintf(file_halo_left, "halo_left.%d", this->topology.get_rank());
    sprintf(file_halo_right, "halo_right.%d", this->topology.get_rank());
    sprintf(file_halo_top, "halo_top.%d", this->topology.get_rank());
    sprintf(file_halo_bottom, "halo_bottom.%d", this->topology.get_rank());

    this->left->print_iter_to_file(file_halo_left);
    this->right->print_iter_to_file(file_halo_right);
    this->top->print_iter_to_file(file_halo_top);
    this->bottom->print_iter_to_file(file_halo_bottom);

    return;
}

void Ghost_Zones::print_buff(struct diag_buff_t *buff) {
    extern fstream logfile;
    
    for (int i = 0; i < buff->nrows; i++) {
        for (int j = 0; j < buff->ncols; j++) {
            logfile << buff->buff[i][j] << "  ";
        }
        logfile << endl;
    }

    return;
}
