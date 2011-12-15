#include <mesh2d.h>

Mesh::Mesh(int m, int n) {
    init(m, n, 1, 1);
}

Mesh::Mesh(int m, int n, int num_halos, int numthreads) {
    init(m, n, num_halos, numthreads);
}

void Mesh::init(int m, int n, int num_halos, int numthreads) {
    this->data_u = this->data_v = this->buff_left = this->buff_right = NULL;

    // Instancia a topologia cartesiana
    this->topology = new Topology(2); // 2D

    // Recupera as coordenadas do processo na topologia
    struct coords c = topology->get_coords();
    // Recupera o número de processos em cada dimensão
    struct dims d = topology->get_dims();
    /* Determina os intervalos da malha para o processo (i, j) na dimensao y (vertical) */
    this->set_interval(m, c.y, d.x, &(this->range.start_y), &(this->range.end_y));
    /* Determina os intervalos da malha para o processo (i, j) na dimensao x (horizontal) */
    this->set_interval(n, c.x, d.y, &(this->range.start_x), &(this->range.end_x));
    /* Seta o tamanho de cada dimensao */
    this->range.global_size_x = n;
    this->range.global_size_y = m;
    this->range.size_x = this->range.end_x - this->range.start_x + 1;
    this->range.size_y = this->range.end_y - this->range.start_y + 1;

    if (num_halos > 1) {
        // Cria os vizinhos nas diagonais para transferir os cantos dos subdominios
        this->topology->create_diag_neighbors();
    }

    if (this->topology->has_diagonal()) {
        /* Cria o tipo de dado MPI nao contiguo para transferir os cantos */
        MPI_Type_vector(num_halos * HALO_THICKNESS, // Número de blocos
                num_halos * HALO_THICKNESS, // Tamanho dos blocos em nro de elementos
                this->range.size_x, // Deslocamento entre cada bloco
                MPI_DOUBLE, // Tipo de dado de cada elemento
                &(this->type_corner));

        MPI_Type_commit(&(this->type_corner));
    }

    this->thread_args = new struct args_t[50];
    this->thread_id = new pthread_t[50];
    pthread_attr_init(&this->thread_attr);
    pthread_attr_setdetachstate(&this->thread_attr, PTHREAD_CREATE_JOINABLE);
    
    //// *Debug*
    //Topology_output::print_info((*topology));
    //return;
    //// *Debug*

    struct mesh_t mesh;

    mesh.mesh = &(this->data_source);
    mesh.size_x = this->range.size_x;
    mesh.size_y = this->range.size_y;

    // Instancia a região de sobreposição
    this->ghost_zones = new Ghost_Zones(this->range.size_y, this->range.size_x,
            mesh, *(this->topology), num_halos);
    // Aloca o mesh
    this->alloc_mesh();
    // Aloca o buffer para o MPI_Pack das bordas da esquerda e da direita
    alloc_cont_array2d(&(this->buff_left), num_halos * HALO_THICKNESS, this->range.size_y);
    alloc_cont_array2d(&(this->buff_right), num_halos * HALO_THICKNESS, this->range.size_y);

    //// Calcula os √≠ndices para o processamento do miolo
    // Ignora a primeira e √∫ltima linha do subdom√≠nio (miolo)
    int inner_size_x = this->range.size_x - 2;
    int inner_size_y = this->range.size_y - 2;
    int thread_slice_x = inner_size_x / numthreads;
    int thread_slice_y = inner_size_y / numthreads;
    
    this->thread_args[0].lower_row = 1;
    this->thread_args[0].upper_row = thread_slice_y;
    this->thread_args[0].lower_col = 1;
    this->thread_args[0].upper_col = thread_slice_x;
    this->thread_args[0].mesh = this;
    
    if(inner_size_x%numthreads != 0) {
    	this->thread_args[0].upper_col++;
    }
    if(inner_size_y%numthreads != 0) {
    	this->thread_args[0].upper_row++;
    }
    
    this->set_num_threads(numthreads);
    
    return;
}

int Mesh::get_num_threads() {
	return this->num_threads;
}

void Mesh::set_num_threads(int num_threads) {
    if(num_threads == this->num_threads && num_threads > 0) {
    	return;
    }
	
    this->num_threads = num_threads;
    
	//// Calcula os √≠ndices para o processamento do miolo
    // Ignora a primeira e √∫ltima linha do subdom√≠nio (miolo)
    int inner_size_x = this->range.size_x - 2;
    int inner_size_y = this->range.size_y - 2;
    int thread_slice_x = inner_size_x / num_threads;
    int thread_slice_y = inner_size_y / num_threads;
    
    this->thread_args[0].lower_row = 1;
    this->thread_args[0].upper_row = thread_slice_y;
    this->thread_args[0].lower_col = 1;
    this->thread_args[0].upper_col = thread_slice_x;
    
    if(inner_size_x%num_threads != 0) {
    	this->thread_args[0].upper_col++;
    }
    if(inner_size_y%num_threads != 0) {
    	this->thread_args[0].upper_row++;
    }
    
    
    //cout<< "inner_size_x = " << inner_size_x << endl;
    //cout<< "inner_size_y = " << inner_size_y << endl;
    //cout<< "thread_slice_x = " << thread_slice_x << endl;
    //cout<< "thread_slice_y = " << thread_slice_y << endl;
    //cout<< "inner_size_x%num_threads = " << inner_size_x%num_threads << endl;
    //cout<< "inner_size_y%num_threads = " << inner_size_y%num_threads << endl << endl;

    //cout<< "Thread 0: " << endl;
    //cout<< "  lower_row = " << this->thread_args[0].lower_row << endl; 
    //cout<< "  upper_row = " << this->thread_args[0].upper_row << endl;
    //cout<< "  lower_col = " << this->thread_args[0].lower_col << endl;
    //cout<< "  upper_col = " << this->thread_args[0].upper_col << endl;
    
    // deixar isso gen√©rico para um n√∫mero arbitr√°rio de threads
    for(int i = 1; i < num_threads; i++) {
    	this->thread_args[i].lower_row = this->thread_args[i-1].upper_row + 1;
    	this->thread_args[i].upper_row = this->thread_args[i-1].upper_row + thread_slice_y;
        this->thread_args[i].lower_col = this->thread_args[i-1].upper_col + 1;
        this->thread_args[i].upper_col = this->thread_args[i-1].upper_col + thread_slice_x;
        this->thread_args[i].mesh = this;
        
        if(i < (inner_size_x%num_threads)) {
        	this->thread_args[i].upper_col++;
        }
    	if(i < (inner_size_y%num_threads)) {
    		this->thread_args[i].upper_row++;
    	}
        //cout<< "Thread " << i << ":" << endl;
        //cout<< "  lower_row = " << this->thread_args[i].lower_row << endl; 
        //cout<< "  upper_row = " << this->thread_args[i].upper_row << endl;
        //cout<< "  lower_col = " << this->thread_args[i].lower_col << endl;
        //cout<< "  upper_col = " << this->thread_args[i].upper_col << endl;
    }
    
    //// Calcula os √≠ndices para o processamento das bordas
    inner_size_x = this->range.size_x;
    inner_size_y = this->range.size_y;
    thread_slice_x = inner_size_x / num_threads;
    thread_slice_y = inner_size_y / num_threads;

    
    this->thread_args[0].border_lower_row = 0;
    this->thread_args[0].border_upper_row = thread_slice_y-1;
    this->thread_args[0].border_lower_col = 0;
    this->thread_args[0].border_upper_col = thread_slice_x-1;

    
    if(inner_size_x%num_threads != 0) {
    	this->thread_args[0].border_upper_col++;
    }
    if(inner_size_y%num_threads != 0) {
    	this->thread_args[0].border_upper_row++;
    }
    
    //cout<< "inner_size_x = " << inner_size_x << endl;
    //cout<< "inner_size_y = " << inner_size_y << endl;
    //cout<< "thread_slice_x = " << thread_slice_x << endl;
    //cout<< "thread_slice_y = " << thread_slice_y << endl;
    //cout<< "inner_size_x%num_threads = " << inner_size_x%num_threads << endl;
    //cout<< "inner_size_y%num_threads = " << inner_size_y%num_threads << endl << endl;
//    
    //cout<< "Thread 0: " << endl;
    //cout<< "  border_lower_row = " << this->thread_args[0].border_lower_row << endl; 
    //cout<< "  border_upper_row = " << this->thread_args[0].border_upper_row << endl;
    //cout<< "  border_lower_col = " << this->thread_args[0].border_lower_col << endl;
    //cout<< "  border_upper_col = " << this->thread_args[0].border_upper_col << endl;
    
    // deixar isso gen√©rico para um n√∫mero arbitr√°rio de threads
    for(int i = 1; i < num_threads; i++) {
    	this->thread_args[i].border_lower_row = this->thread_args[i-1].border_upper_row + 1;
    	this->thread_args[i].border_upper_row = this->thread_args[i-1].border_upper_row + thread_slice_y;
        this->thread_args[i].border_lower_col = this->thread_args[i-1].border_upper_col + 1;
        this->thread_args[i].border_upper_col = this->thread_args[i-1].border_upper_col + thread_slice_x;
        
        if(i < (inner_size_x%num_threads)) {
        	this->thread_args[i].border_upper_col++;
        }
    	if(i < (inner_size_y%num_threads)) {
    		this->thread_args[i].border_upper_row++;
    	}
        //cout<< "Thread " << i << ":" << endl;
        //cout<< "  border_lower_row = " << this->thread_args[i].border_lower_row << endl; 
        //cout<< "  border_upper_row = " << this->thread_args[i].border_upper_row << endl;
        //cout<< "  border_lower_col = " << this->thread_args[i].border_lower_col << endl;
        //cout<< "  border_upper_col = " << this->thread_args[i].border_upper_col << endl;
    }
}

Mesh::~Mesh() {
    this->free_mesh();
}

void Mesh::alloc_mesh(void) {
    if (this->data_u != NULL || this->data_v != NULL) {
        return;
    }

    alloc_cont_array2d(&(this->data_u), this->range.size_y, this->range.size_x);
    alloc_cont_array2d(&(this->data_v), this->range.size_y, this->range.size_x);

    this->data_source = this->data_u;
    this->data_dest = this->data_v;

    return;
}

void Mesh::free_mesh(void) {
    /* Não libera aquilo que nao foi alocado */
    if (this->data_u != NULL || this->data_v != NULL) {
        return;
    }

    /* Libera o subdominio */
    /* Nao precisa fazer um laço de free porque a matriz eh alocada
    de forma contigua em memoria (ver funcao alloc_array2d) */
    free(this->data_u[0]);
    free(this->data_u);

    free(this->data_v[0]);
    free(this->data_v);

    return;
}

void Mesh::set_interval(int size, int rank, int nprocs, int *start, int *end) {
    int blk_size = size / nprocs;
    int blk_remainder = size % nprocs;
    int padding = 0;

    if (blk_remainder > 0) {
        if (rank < blk_remainder) {
            padding = rank;
        } else {
            padding = blk_remainder;
        }
    }

    *start = rank * blk_size + padding;
    *end = *start + blk_size;
    if (rank >= blk_remainder) {
        *end = *end - 1;
    }

    return;
}

double& Mesh::operator()(int row, int col) {
    return this->data_dest[row][col];
}

double Mesh::operator()(int row, int col) const {
    return this->data_dest[row][col];
}

/*
 * Além do data_souce, possibilita acesso aos dados do halo
 */
double Mesh::get_source_halo(int row, int col) {
    if (row < 0) {
        return (*(this->ghost_zones->top))[col];
    } else if (row >= this->get_size_y()) {
        return (*(this->ghost_zones->bottom))[col];
    } else if (col < 0) {
        return (*(this->ghost_zones->left))[row];
    } else if (col >= this->get_size_x()) {
        return (*(this->ghost_zones->right))[row];
    } else {
        return this->data_source[row][col];
    }

    return 0.0;
}

/*
 * Troca o endereço entre data_source e data_dest
 */
void Mesh::swap(void) {
    double **temp = this->data_source;

    this->data_source = this->data_dest;
    this->data_dest = temp;

    return;
}

int Mesh::get_size_x(void) {
    return this->range.size_x;
}

int Mesh::get_size_y(void) {
    return this->range.size_y;
}

int Mesh::get_begin_x(void) {
    return this->range.start_x;
}

int Mesh::get_end_x(void) {
    return this->range.end_x;
}

int Mesh::get_begin_y(void) {
    return this->range.start_y;
}

int Mesh::get_end_y(void) {
    return this->range.end_y;
}

/*
 * Inicia o envio das fronteiras
 */
void Mesh::send_borders(void) {
    int pos = 0;

    // Inicializa o update das ghost_zones
    this->transfer = this->ghost_zones->update();

    if (this->transfer) {
        // Obtém os iteradores dos vizinhos
        neighbors_iterator neighbor = this->topology->first_neighbor();
        neighbors_iterator last_neighbor = this->topology->last_neighbor();

        int num_halos = this->ghost_zones->get_num_halos();

        // Itera os vizinhos
        while (neighbor != last_neighbor) {
            switch ((*neighbor).direction) {
                case RIGHT: /* Envia sua fronteira para seu vizinho da direita */
                    pos = 0;
                    for (int i = this->get_size_x() - (num_halos * HALO_THICKNESS); i < this->get_size_x(); i++) {
                        for (int j = 0; j < this->get_size_y(); j++) {
                            MPI_Pack(&this->data_source[j][i], 1, MPI_DOUBLE,
                                    this->buff_right[0], num_halos * HALO_THICKNESS * this->get_size_y() * sizeof (double),
                                    &pos, this->topology->get_comm());
                        }
                    }

                    MPI_Isend(this->buff_right[0], // buff
                            pos, // count
                            MPI_PACKED, // datatype
                            (*neighbor).rank, // dest
                            TO_LEFT_NEIGHBOR_HALO_TAG, // tag
                            this->topology->get_comm(), // comm
                            &(this->sreq_right)); // request
                    break;
                case LEFT: /* Envia sua fronteira para seu vizinho da direita */
                    pos = 0;
                    for (int i = 0; i < num_halos; i++) {
                        for (int j = 0; j < this->get_size_y(); j++) {
                            MPI_Pack(&this->data_source[j][i], 1, MPI_DOUBLE,
                                    this->buff_left[0], num_halos * HALO_THICKNESS * this->get_size_y() * sizeof (double),
                                    &pos, this->topology->get_comm());
                        }
                    }

                    MPI_Isend(this->buff_left[0], // buff
                            pos, // count
                            MPI_PACKED, // datatype
                            (*neighbor).rank, // dest
                            TO_RIGHT_NEIGHBOR_HALO_TAG, // tag
                            this->topology->get_comm(), // comm
                            &(this->sreq_left)); // request
                    break;
                case TOP: /* Envia sua fronteira para seu vizinho de cima */
                    MPI_Isend(this->data_source[0],
                            this->get_size_x() * num_halos * HALO_THICKNESS,
                            MPI_DOUBLE,
                            (*neighbor).rank,
                            TO_BOTTOM_NEIGHBOR_HALO_TAG,
                            this->topology->get_comm(),
                            &(this->sreq_top));
                    break;
                case BOTTOM: /* Envia sua fronteira para seu vizinho de baixo */
                    MPI_Isend(this->data_source[this->get_size_y() - (num_halos * HALO_THICKNESS)],
                            this->get_size_x() * num_halos * HALO_THICKNESS,
                            MPI_DOUBLE,
                            (*neighbor).rank,
                            TO_TOP_NEIGHBOR_HALO_TAG,
                            this->topology->get_comm(),
                            &(this->sreq_bottom));
                    break;
                case TOP_LEFT:
                    MPI_Isend(this->data_source[0],
                            1,
                            this->type_corner,
                            (*neighbor).rank,
                            TO_TOP_LEFT_NGBR_TAG,
                            this->topology->get_comm(),
                            &(this->sreq_top_left));
                    break;
                case TOP_RIGHT:
                    MPI_Isend(this->data_source[0] + this->get_size_x() - (num_halos * HALO_THICKNESS),
                            1,
                            this->type_corner,
                            (*neighbor).rank,
                            TO_TOP_RIGHT_NGBR_TAG,
                            this->topology->get_comm(),
                            &(this->sreq_top_right));
                    break;
                case BOTTOM_LEFT:
                    MPI_Isend(this->data_source[this->get_size_y() - (num_halos * HALO_THICKNESS)],
                            1,
                            this->type_corner,
                            (*neighbor).rank,
                            TO_BOTTOM_LEFT_NGBR_TAG,
                            this->topology->get_comm(),
                            &(this->sreq_bottom_left));
                    break;
                case BOTTOM_RIGHT:
                    MPI_Isend(this->data_source[this->get_size_y() - (num_halos * HALO_THICKNESS)] + this->get_size_x() - (num_halos * HALO_THICKNESS),
                            1,
                            this->type_corner,
                            (*neighbor).rank,
                            TO_BOTTOM_RIGHT_NGBR_TAG,
                            this->topology->get_comm(),
                            &(this->sreq_bottom_right));
                    break;
                default:
                    cout << "Huh? " << (*neighbor).direction << endl;
                    break;
            }
            // Próximo vizinho
            neighbor++;
        }
    }

    return;
}

/*
 * Espera pelos Isends
 */
void Mesh::sync(void) {
    if (this->transfer) {
        // Obtém os iteradores dos vizinhos
        neighbors_iterator neighbor = this->topology->first_neighbor();
        neighbors_iterator last_neighbor = this->topology->last_neighbor();
        // Itera os vizinhos
        while (neighbor != last_neighbor) {
            switch ((*neighbor).direction) {
                case RIGHT:
                    MPI_Wait(&(this->sreq_right), MPI_STATUS_IGNORE);
                    break;
                case LEFT:
                    MPI_Wait(&(this->sreq_left), MPI_STATUS_IGNORE);
                    break;
                case TOP:
                    MPI_Wait(&(this->sreq_top), MPI_STATUS_IGNORE);
                    break;
                case BOTTOM:
                    MPI_Wait(&(this->sreq_bottom), MPI_STATUS_IGNORE);
                    break;
                case TOP_LEFT:
                    MPI_Wait(&(this->sreq_top_left), MPI_STATUS_IGNORE);
                    break;
                case TOP_RIGHT:
                    MPI_Wait(&(this->sreq_top_right), MPI_STATUS_IGNORE);
                    break;
                case BOTTOM_LEFT:
                    MPI_Wait(&(this->sreq_bottom_left), MPI_STATUS_IGNORE);
                    break;
                case BOTTOM_RIGHT:
                    MPI_Wait(&(this->sreq_bottom_right), MPI_STATUS_IGNORE);
                    break;
                default:
                    cout << "Huh?  " << (*neighbor).direction << endl;
                    break;
            }
            // Próximo vizinho
            neighbor++;
        }
    }
    
    this->ghost_zones->sync();

    return;
}

/*
 * Inicializa a halo externa mais a esquerda de todo o mesh com um valor que será
 * propagado
 */
void Mesh::set_left_extern(double value) {
    // Se o vizinho a esquerda for MPI_PROC_NULL, então a halo é externa
    if (this->ghost_zones->left->get_neighbor_rank() == MPI_PROC_NULL) {
        for (int i = 0; i < this->get_size_y(); i++) {
        	(*(this->ghost_zones->left))[i] = value;
        }
    }

    return;
}

/*
 * Inicializa a halo externa mais a direita do mesh com um valor que será
 * propagado
 */
void Mesh::set_right_extern(double value) {
    // Se o vizinho a direita for MPI_PROC_NULL, então a halo é externa
    if (this->ghost_zones->right->get_neighbor_rank() == MPI_PROC_NULL) {
        for (int i = 0; i < this->get_size_y(); i++) {
        	(*(this->ghost_zones->right))[i] = value;
        }
    }

    return;
}

/*
 * Inicializa a halo externa mais acima do mesh com um valor que será
 * propagado
 */
void Mesh::set_top_extern(double value) {
    // Se o vizinho de cima for MPI_PROC_NULL, então a halo é externa
    if (this->ghost_zones->top->get_neighbor_rank() == MPI_PROC_NULL) {
        for (int i = 0; i < this->get_size_x(); i++) {
        	(*(this->ghost_zones->top))[i] = value;
        }
    }

    return;
}

/*
 * Inicializa a halo externa mais abaixo do mesh com um valor que será
 * propagado
 */
void Mesh::set_bottom_extern(double value) {
    // Se o vizinho de baixo for MPI_PROC_NULL, então a halo é externa
    if (this->ghost_zones->bottom->get_neighbor_rank() == MPI_PROC_NULL) {
        for (int i = 0; i < this->get_size_x(); i++) {
        	(*(this->ghost_zones->bottom))[i] = value;
        }
    }

    return;
}

void Mesh::gather(void) {
    int proc_ranges[4], size_x, size_y;
    double **temp = NULL;
    MPI_Status s;

    if (this->topology->get_rank() == 0) {
        alloc_cont_array2d(&(this->final_result), this->range.global_size_y, this->range.global_size_x);

        copy_vals(&(this->data_source), this->get_begin_x(), this->get_end_x(),
                this->get_begin_y(), this->get_end_y());


        /* Recebe de cada processo */
        for (int i = 1; i < this->topology->get_nprocs(); i++) {
            /* Recebe os intervalos do processo */
            MPI_Recv(proc_ranges, 4, MPI_INT, i, PROC_RANGES_RESULT_TAG, this->topology->get_comm(), &s);
            size_x = proc_ranges[1] - proc_ranges[0] + 1;
            size_y = proc_ranges[3] - proc_ranges[2] + 1;

            /* Aloca o buffer para receber a parte do processo */
            alloc_cont_array2d(&temp, size_y, size_x);
            MPI_Recv(temp[0], size_y * size_x, MPI_DOUBLE, i, PROC_RESULT_TAG,
                    this->topology->get_comm(), &s);
            /* Copia os valores recebidos para o resultado final */
            copy_vals(&temp, proc_ranges[0], proc_ranges[1], proc_ranges[2], proc_ranges[3]);

            free(temp[0]);
            free(temp);
        }
    } else {
        proc_ranges[0] = this->get_begin_x();
        proc_ranges[1] = this->get_end_x();
        proc_ranges[2] = this->get_begin_y();
        proc_ranges[3] = this->get_end_y();

        MPI_Send(proc_ranges, 4, MPI_INT, 0, PROC_RANGES_RESULT_TAG, this->topology->get_comm());
        MPI_Send(this->data_source[0], this->get_size_x() * this->get_size_y(),
                MPI_DOUBLE, 0, PROC_RESULT_TAG, this->topology->get_comm());
    }

    return;
}

/* Copia os valores de vals para as linhas de start_y até end_y e colunas
start_x até end_x da matriz this->final_result */
void Mesh::copy_vals(double ***vals, int start_x, int end_x, int start_y, int end_y) {
    int l = start_y, c = start_x;
    int size_x = end_x - start_x + 1;
    int size_y = end_y - start_y + 1;

    for (int i = 0; i < size_y; i++) {
        c = start_x;
        for (int j = 0; j < size_x; j++) {
            this->final_result[l][c++] = (*vals)[i][j];
        }
        l++;
    }

    return;
}

void Mesh::print_final_result(void) {
    if (this->topology->get_rank() != 0) {
        return;
    }

    char filename[50];

    sprintf(filename, "result.%dx%d", this->range.global_size_x, this->range.global_size_y);

    ifstream ifile(filename);
    if (ifile) {
        return;
    }

    fstream fout(filename, ios::out);

    fout << fixed << setprecision(6);

    for (int i = 0; i < this->range.global_size_y; i++) {
        for (int j = 0; j < this->range.global_size_x; j++) {
            fout << this->final_result[i][j] << "\t";
        }
        fout << endl;
    }

    fout.close();

    return;
}

void Mesh::print_mesh_info(void) {
    char filename[50];

    sprintf(filename, "mesh.%d", this->topology->get_rank());

    fstream fout(filename, ios::out);

    fout << "global_size_x = " << this->range.global_size_x << endl;
    fout << "global_size_y = " << this->range.global_size_y << endl;
    fout << "size_x = " << this->range.size_x << endl;
    fout << "size_y = " << this->range.size_y << endl;
    fout << "start_x = " << this->range.start_x << endl;
    fout << "end_x   = " << this->range.end_x << endl;
    fout << "start_y = " << this->range.start_y << endl;
    fout << "end_y   = " << this->range.end_y << endl;

    fout.close();

    return;
}

void Mesh::print_file_mesh(void) {
    if (this->data_u == NULL || this->data_v == NULL) {
        return;
    }

    char filename[50];

    sprintf(filename, "mesh_data.%d", this->topology->get_rank());

    fstream fout(filename, ios::out);

    fout << fixed << setprecision(6);

    fout << "data_source:" << endl;
    for (int i = 0; i < this->range.size_y; i++) {
        for (int j = 0; j < this->range.size_x; j++) {
            fout << this->data_source[i][j] << "\t";
        }
        fout << endl;
    }

    fout << endl << "data_dest:" << endl;
    for (int i = 0; i < this->range.size_y; i++) {
        for (int j = 0; j < this->range.size_x; j++) {
            fout << this->data_dest[i][j] << "\t";
        }
        fout << endl;
    }

    fout.close();
}

void Mesh::print_buff(fstream &dbg, double **buff, int start_row, int num_rows, int start_col, int num_cols) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            dbg << buff[i + start_row][j + start_col] << "  ";
        }
        dbg << endl;
    }

    return;
}
