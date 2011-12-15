#include <topology.h>

Topology::Topology(int ndims, int stencil_type, int periods[], int reorder, MPI_Comm comm_old) {
    this->periods = (int*)calloc(ndims, sizeof(int));
    
    if(periods != NULL){
        /*     dest           src */
        memcpy(this->periods, periods, ndims);
    }

    this->init(comm_old, stencil_type, ndims, this->periods, reorder);
    
    return;
}

int Topology::init(MPI_Comm comm_old, int stencil_type, int ndims, int periods[], int reorder){
    /* Nao existe dimensao <= 0 */
    if(ndims <= 0){
        return(1);
    }

    int comm_old_nprocs = 0;
    int dims[ndims];       /* Numero de processos em cada dimensao */
    int coords[ndims];     /* Coordenadas (x,y,z) do processo na topologia */

    this->has_diag = false;

    /* Inicializa o numero de dimensoes na estrutura */
    this->set_ndims(ndims);
    /* Inicializa o vetor de dimensoes com 0 */
    for(int i = 0; i < ndims; i++){
        dims[i] = 0;
    }
    
    /* Pega o tamanho do comunicador a partir do qual a topologia sera criada */
    if(MPI_Comm_size(comm_old, &comm_old_nprocs) != MPI_SUCCESS){
        return(1);
    }
    
    /* Determina o numero de processos em cada dimensao */
    if(MPI_Dims_create(comm_old_nprocs, ndims, dims) != MPI_SUCCESS){
        return(1);
    }
    
    this->set_dims(dims);
    
    /* Cria a topologia cartesiana */
    if(MPI_Cart_create(comm_old, ndims, dims, periods, reorder, &(this->comm)) != MPI_SUCCESS){
        return(1);
    }
    /* Pega o rank do processo no comunicador criado a partir da topologia */
    if(MPI_Comm_rank(this->comm, &(this->rank)) != MPI_SUCCESS){
        return(1);
    }
    /* Pega o numero de processos no comunicador da topologia */
    if(MPI_Comm_size(this->comm, &(this->nprocs)) != MPI_SUCCESS){
        return(1);
    }

    /* Pega as coordenadas do processo */
    if(MPI_Cart_coords(this->comm, this->rank, ndims, coords) != MPI_SUCCESS){
        return(1);
    }
    
    this->set_coords(coords);

    /* Determina o rank dos processos vizinhos */
    if(meet_the_neighbors()){
        return(1);
    }
    
    if (stencil_type == STAR_TYPE_STENCIL) {
        this->create_diag_neighbors();
    }
    
    return(0);
}

Topology::~Topology(void){
    free(this->periods);
    delete this->neighbors;
}

/**
 *   Método que determina os vizinhos do processo na topologia cartesiana
 */
int Topology::meet_the_neighbors(void){
    int n_x[2] = {MPI_PROC_NULL, MPI_PROC_NULL}; 
    int n_y[2] = {MPI_PROC_NULL, MPI_PROC_NULL}; 
    int n_z[2] = {MPI_PROC_NULL, MPI_PROC_NULL};
    vector<struct neighbor> n;
    struct neighbor neighbor;

    switch(this->get_ndims()) {
        case 3:
            if(MPI_Cart_shift(this->comm, 2, 1, &(n_z[0]), &(n_z[1])) != MPI_SUCCESS){
                return(3);
            }
            if(n_z[0] != MPI_PROC_NULL){
                neighbor.rank = n_z[0]; // Front
                neighbor.direction = FRONT;
                n.push_back(neighbor);
            }

            if(n_z[1] != MPI_PROC_NULL){
                neighbor.rank = n_z[1]; // Back
                neighbor.direction = BACK;
                n.push_back(neighbor);
            }
        case 2:
            if(MPI_Cart_shift(this->comm, 0, 1, &(n_y[0]), &(n_y[1])) != MPI_SUCCESS){
                return(2);
            }
            
            if(n_y[0] != MPI_PROC_NULL){
                neighbor.rank = n_y[0]; // Top
                neighbor.direction = TOP;
                n.push_back(neighbor);
            }

            if(n_y[1] != MPI_PROC_NULL){
                neighbor.rank = n_y[1]; // Bottom
                neighbor.direction = BOTTOM;
                n.push_back(neighbor);
            }
        case 1:
            if(MPI_Cart_shift(this->comm, 1, 1, &(n_x[0]), &(n_x[1])) != MPI_SUCCESS){
                return(1);
            }

            if(n_x[0] != MPI_PROC_NULL){
                neighbor.rank = n_x[0]; // Left
                neighbor.direction = LEFT;
                n.push_back(neighbor);
            }

            if(n_x[1] != MPI_PROC_NULL){
                neighbor.rank = n_x[1]; // Right
                neighbor.direction = RIGHT;
                n.push_back(neighbor);
            }
            break;
        default:
            return(4);
    }
    
    // Número total de vizinhos + 1. A última entrada demarca o fim da lista
    // de vizinhos
    this->num_neighbors = n.size() + 1;
    this->neighbors = new struct neighbor[this->num_neighbors];

    vector<struct neighbor>::iterator it = n.begin();
    int i = 0;
    
    while(it != n.end()){
        this->neighbors[i].rank = (*it).rank;
        this->neighbors[i].direction = (*it).direction;
        
        it++; i++;
    }
    
    // Marca de fim de lista
    this->neighbors[this->num_neighbors-1].rank = -1;
    this->neighbors[this->num_neighbors-1].direction = -1;
    
    return(0);
}

bool Topology::has_diagonal(void) {
    return this->has_diag;
}

void Topology::create_diag_neighbors(void) {
    struct dims nprocs_dims = this->get_dims();
    int nprocs_y = nprocs_dims.y;
    int num_neighbors_new = this->num_neighbors;

    // Se o nro de processos for primo, então não há diagonais
    if(this->dims.x == 1 || this->dims.y == 1) {
        this->has_diag = false;
        return;
    }

    this->has_diag = true;

    // Dependendo do nro de vizinhos, incrementa o nro de vizinhos
    // nas diagonais
    switch (num_neighbors_new-1) {
        case 2: // Caso tenha 2 vizinhos, possui +1 vizinho na diagonal
            num_neighbors_new += 1;
            break;
        case 3:
            num_neighbors_new += 2;
            break;
        case 4:
            num_neighbors_new += 4;
            break;
        default:
            return;
    }
    
    // Recupera os iteradores dos vizinhos para descobrir
    // a posição vertical do processo atual na topologia.
    // A posição vertical se refere ao fato do processo estar ou no meio
    // da topologia, ou na parte superior, ou ainda na parte inferior.
    // Os iteradores são utilizados também para recriar o vetor que
    // armazena os dados dos vizinhos, incluíndo os novos vizinhos das
    // diagonais.
    neighbors_iterator curr = this->first_neighbor();
    neighbors_iterator last = this->last_neighbor();
    
    // Nova estrutura que substituirá àquela criada pelo método "meet_the_neighbors"
    struct neighbor *n = new struct neighbor[num_neighbors_new];
    // Iterador para o controle do índice atual do vetor n
    int i = 0;
    // Define a localização do processo na topologia. Pode ser LOCATION_TOP,
    // LOCATION_BOTTOM ou LOCATION_MIDDLE
    int location = 0; 
    
    // Itera todos os vizinhos atuais.
    while (curr != last) {
        // Transfere os dados dos vizinhos para a nova estrutura
        n[i].rank = (*curr).rank;
        n[i].direction = (*curr).direction;
        
        // Se houver apenas vizinho acima...
        if (n[i].direction == TOP) {
            // ...então o processo está na parte mais inferior da topologia.
            location |= LOCATION_BOTTOM;
        // Entretanto, se também existir vizinho abaixo...
        } else if (n[i].direction == BOTTOM) {
            // ...então o processo está no meio da topologia. LOCATION_MIDDLE == (LOCATION_BOTTOM | LOCATION_TOP)
            location |= LOCATION_TOP;
        }
        
        curr++; i++;
    }
    
    // Novo iterador para definir os processos das diagonais
    neighbors_iterator it = this->first_neighbor();
    
    // Se o processo está no meio da topologia, então ele terá quatro novos
    // vizinhos nas diagonais. Para definí-los, o algoritmo considera apenas
    // os vizinhos das laterais (LEFT e RIGHT). Isso vale para o caso dele ser
    // LOCATION_TOP e LOCATION_BOTTOM.
    if(location == LOCATION_MIDDLE) {
        while(it != last) {
            // Considera apenas os processos das laterais (LEFT e RIGHT)
            switch ((*it).direction) {
                // Se existir processo à esquerda, define as diagonais a esquerda
                case LEFT:
                    // Define o rank do processo acima e a esquerda
                    n[i].rank = this->rank - nprocs_y - 1;
                    n[i++].direction = TOP_LEFT;
                    // Define o rank do processo abaixo e a esquerda
                    n[i].rank = this->rank + nprocs_y - 1;
                    n[i++].direction = BOTTOM_LEFT;
                    break;
                case RIGHT:
                    n[i].rank = this->rank - nprocs_y + 1;
                    n[i++].direction = TOP_RIGHT;
                    n[i].rank = this->rank + nprocs_y + 1;
                    n[i++].direction = BOTTOM_RIGHT;
                    break;
                default:
                    break;
            }
            it++;
        }
    // Caso o processo esteja no topo
    } else if (location == LOCATION_TOP) {
        while(it != last) {
            switch ((*it).direction) {
                // Para o caso de existir um processo à esquerda, define o processo
                // da diagonal de baixo e da esquerda
                case LEFT:
                    n[i].rank = this->rank + nprocs_y - 1;
                    n[i++].direction = BOTTOM_LEFT;
                    break;
                case RIGHT:
                    n[i].rank = this->rank + nprocs_y + 1;
                    n[i++].direction = BOTTOM_RIGHT;
                    break;
                default:
                    break;
            }
            it++;
        }
    // Caso o processo esteja na parte inferior
    } else if (location == LOCATION_BOTTOM) {
        while(it != last) {
            switch ((*it).direction) {
                // Para o caso de existir um processo à esquerda, define o processo
                // da diagonal de cima e da esquerda
                case LEFT:
                    n[i].rank = this->rank - nprocs_y - 1;
                    n[i++].direction = TOP_LEFT;
                    break;
                case RIGHT:
                    n[i].rank = this->rank - nprocs_y + 1;
                    n[i++].direction = TOP_RIGHT;
                    break;
                default:
                    break;
            }
            it++;
        }
    }
    
    // Marca o fim da lista de vizinhos
    n[i].rank = -1;
    n[i].direction = -1;
    
    // Substitui o vetor de vizinhos do objeto pelo novo vetor criado, que
    // contém os novos vizinhos das diagonais
    delete this->neighbors;
    this->neighbors = n;
    this->num_neighbors = num_neighbors_new;
    
    return;
}

neighbors_iterator Topology::first_neighbor(void) {
    return neighbors_iterator(this->neighbors);
}

neighbors_iterator Topology::last_neighbor(void) {
    return neighbors_iterator(&(this->neighbors[this->num_neighbors - 1]));
}

int Topology::get_rank(void){
    return this->rank;
}

int Topology::get_ndims(void){
    return this->ndims;
}

MPI_Comm Topology::get_comm(void){
    return this->comm;
}

struct dims Topology::get_dims(void){
    return this->dims;
}

struct coords Topology::get_coords(void){
    return this->coords;
}

void Topology::set_ndims(int ndims){
    this->ndims = ndims;
    
    return;
}

void Topology::set_coords(int coords[]){
    this->coords.x = -1;
    this->coords.y = -1;
    this->coords.z = -1;

    switch(this->get_ndims()){
        case 3:
            this->coords.z = coords[2];
        case 2:
            this->coords.y = coords[0];
        case 1:
            this->coords.x = coords[1];
        default:
            break;
    }
    
    return;
}

void Topology::set_dims(int dims[]){
    this->dims.x = -1;
    this->dims.y = -1;
    this->dims.z = -1;

    switch(this->get_ndims()){
        case 3:
            this->dims.z = dims[2];
        case 2:
            this->dims.y = dims[1];
        case 1:
            this->dims.x = dims[0];
        default:
            break;
    }
    
    return;    
}

int Topology::get_nprocs(void){
    return this->nprocs;
}

