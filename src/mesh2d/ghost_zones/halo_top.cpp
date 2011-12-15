#include <halo.h>

void* jacobi2d_top(void *args) {
    Halo_Top *This = (Halo_Top*) args;

    for (int halo = This->num_iterators_ctrl; halo < This->num_iterators; halo++) {
        for (halo_iterator hs = This->h_source[halo]->first(),
                hd = This->h_dest[halo]->first();; hs++, hd++) {
            *hd = hs.reduce() * 0.25;

            if (hs == This->h_source[halo]->last()) {
                break;
            }
        }
    }

    halo_iterator hd = This->h_dest[0]->first();

    for (int i = 0; i < This->mesh.size_y; i++, hd++) {
        *hd = *hd + ((*(This->mesh.mesh))[0][i] * 0.25);
    }

    pthread_exit(NULL);
}

/**
 * \brief Construtor do Halo_Top, sobrepondo-o com o vizinho de rank rank_neighbor.
 * 
 * @param [in] halo_size      Tamanho do Halo_Top.
 * @param [in] num_halos      Número de Halos.
 * @param [in] rank_neighbor  Rank do vizinho com o qual ele sobrepõe-se.
 * @param [in] direction_tag  Tag utilizado para receber a borda do vizinho.
 * @param [in] comm           Comunicador MPI.
 * @return
 */
Halo_Top::Halo_Top(int halo_size, int num_halos, struct mesh_t mesh,
        struct diag_buff_t *corner_left, struct diag_buff_t *corner_right,
        int rank_neighbor, int direction_tag, MPI_Comm comm) {

    this->num_rows = num_halos * HALO_THICKNESS + (2 * HALO_THICKNESS);
    this->num_cols = halo_size + (2 * HALO_THICKNESS);

    this->start_row = HALO_THICKNESS;
    this->end_row = this->num_rows - HALO_THICKNESS - 1;
    this->start_col = HALO_THICKNESS;
    this->end_col = this->num_cols - HALO_THICKNESS - 1;

    this->mesh = mesh;

    this->corner_left.buff = corner_left->buff;
    this->corner_left.start_row = 0;
    this->corner_left.end_row = corner_left->nrows - 1;
    this->corner_left.start_col = corner_left->ncols - HALO_THICKNESS;
    this->corner_left.end_col = corner_left->ncols - 1;
    this->corner_left.nrows = corner_left->nrows;
    this->corner_left.ncols = corner_left->ncols;
    this->corner_left.rank_source = corner_left->rank_source;

    this->corner_right.buff = corner_right->buff;
    this->corner_right.start_row = 0;
    this->corner_right.end_row = corner_right->nrows - 1;
    this->corner_right.start_col = 0;
    this->corner_right.end_col = HALO_THICKNESS - 1;
    this->corner_right.nrows = corner_right->nrows;
    this->corner_right.ncols = corner_right->ncols;
    this->corner_right.rank_source = corner_right->rank_source;

    pthread_attr_init(&this->thread_attr);
    pthread_attr_setdetachstate(&this->thread_attr, PTHREAD_CREATE_JOINABLE);

    // Tamanho do Halo_Top.
    this->halo_size = halo_size;
    // Número de Halos.
    this->num_halos = num_halos;
    // O vizinho é nulo. Pode ser alterado mais tarde por meio da chamada
    // do mesmo método.
    this->set_neighbor_info(rank_neighbor, direction_tag, comm);

    this->data_u = this->data_v = this->data_source = this->data_dest = NULL;
    this->buff = NULL;
    this->h_source = this->h_dest = NULL;

    // Aloca o Halo_Top.
    this->alloc();
    // Define os iteradores.
    this->set_iterators();
}

/**
 * \brief Destrutor.
 * @return
 */
Halo_Top::~Halo_Top() {
    // Libera o vetor.
    if (this->data_source != NULL) {
        free_cont_array2d(&(this->data_source));
    }
    if (this->data_dest != NULL) {
        free_cont_array2d(&(this->data_dest));
    }

    // Liberar os iteradores h_source e h_dest

    this->data_source = this->data_dest = this->data_u = this->data_v = NULL;
}

/**
 * \brief Aloca o vetor bidimensional que conterá os dados do Halo_Top.
 * @return Primeira posição de memória do espaço alocado.
 */
void Halo_Top::alloc(void) {

    alloc_cont_array2d(&(this->data_u), this->num_rows, this->num_cols);
    // Se tiver mais de uma região de sobreposição,
    // aloca o array2d adicional para possibilitar o cálculo
    if (this->get_num_halos() > 1) {
        alloc_cont_array2d(&(this->data_v), this->num_rows, this->num_cols);
    }

    this->data_source = this->data_u;
    this->data_dest = this->data_v;

    alloc_cont_array2d(&(this->buff), this->num_halos, this->halo_size);

    return;
}

void Halo_Top::set_iterators(void) {
    if (this->get_num_halos() <= 1) {
        return;
    }

    // Os iteradores devem desconsiderar as linhas e colunas extras do halo,
    // utilizadas apenas para manter as referências do stencil válidas.
    this->num_iterators = this->num_rows - (2 * HALO_THICKNESS);
    this->num_iterators_ctrl = this->num_iterators;

    h_source = new H*[this->num_iterators];
    h_dest = new H*[this->num_iterators];

    for (int i = 0, j = this->start_row; i < this->num_iterators; i++, j++) {
        h_source[i] = new H(&(this->data_source), j, this->start_col, this->get_halo_size(), this->num_cols);
    }

    for (int i = 0, j = this->start_row; i < this->num_iterators; i++, j++) {
        h_dest[i] = new H(&(this->data_dest), j, this->start_col, this->get_halo_size(), this->num_cols);
    }

    return;
}

void Halo_Top::swap(void) {
    double **temp = this->data_source;

    this->data_source = this->data_dest;
    this->data_dest = temp;

    return;
}

/**
 * \brief Sobrepõe o Halo_Top com um vizinho.
 * 
 * @param [in] rank_neighbor  Rank do vizinho.
 * @param [in] direction_tag  Tag da mensagem.
 * @param [in] comm           Comunicador MPI.
 */
void Halo_Top::set_neighbor_info(int rank_neighbor, int direction_tag, MPI_Comm comm) {
    // Rank do vizinho
    if (rank_neighbor < 0) {
        this->rank_neighbor = MPI_PROC_NULL;
    } else {
        this->rank_neighbor = rank_neighbor;
    }
    // Direção de onde esperar a mensagem
    this->direction_tag = direction_tag;
    // Comunicador
    this->comm = comm;

    return;
}

/**
 * \brief Devolve o rank do vizinho.
 * @return Rank do vizinho.
 */
int Halo_Top::get_neighbor_rank(void) {
    return this->rank_neighbor;
}

/**
 * \brief Devolve o tamanho do Halo_Top.
 * @return Tamanho do Halo_Top.
 */
int Halo_Top::get_halo_size(void) {
    return this->halo_size;
}

/**
 * \brief Devolve o número de Halos.
 * @return Número de Halos.
 */
int Halo_Top::get_num_halos(void) {
    return this->num_halos;
}

/**
 * \brief Inicia o recebimento assíncrono da borda do vizinho.
 * 
 * O recebimento é feito por meio da primitiva MPI_Irecv, que espera uma
 * mensagem do vizinho informado ou no momento da instanciação do Halo_Top
 * ou no momento da chamada ao método set_neighbor_info. A mensagem é aguardada
 * com o tag e o comunicador passados por parâmetros.
 */
bool Halo_Top::init_update(void) {
    // Não recebe de Halos não sobrepostos.
    if (this->rank_neighbor == MPI_PROC_NULL) {
        return true;
    }

    if (this->transfer()) {
        MPI_Irecv(this->buff[0], // buff
                this->num_halos * this->halo_size, // count
                MPI_DOUBLE, // datatype
                this->rank_neighbor, // source
                this->direction_tag, // tag
                this->comm, // comm
                &(this->request)); // request
        return true;
    } else {
        pthread_create(&(this->thread_id), &(this->thread_attr), jacobi2d_top, this);
        return false;
    }

    return true;
}

bool Halo_Top::transfer(void) {
    if (this->rank_neighbor == MPI_PROC_NULL || this->h_source == NULL) {
        return true;
    }

    return (this->num_iterators_ctrl == this->num_iterators);
}

/**
 * \brief Espera pela conclusão do init_update.
 * 
 * O método invoca a primitiva MPI_Wait a fim de bloquear o processo e garantir
 * a conclusão do MPI_Irecv, disparado pelo init_update.
 */
void Halo_Top::sync(void) {
    // Ignora Halos não sobrepostos.
    if (this->rank_neighbor == MPI_PROC_NULL) {
        return;
    }
    
    if (this->transfer()) {
        // Simula latência de rede
        //usleep(500);
        //////////////////////////
        MPI_Wait(&(this->request), MPI_STATUS_IGNORE);

        for (int i = this->start_row, j = 0; i <= this->end_row; i++, j++) {
            memcpy(this->data_source[i] + HALO_THICKNESS, this->buff[j],
                    this->halo_size * sizeof (double));
        }

        if (this->num_halos > 1) {
            memcpy(this->data_dest[0], this->data_source[0],
                    this->num_rows * this->num_cols * sizeof (double));

            // Reseta o contador que determina se já é necessário transferir as RS
            this->num_iterators_ctrl = 0;

            if (this->corner_left.buff != NULL) {
                // Copia corner left
                for (int i = this->corner_left.start_row, l = this->start_row;
                        i <= this->corner_left.end_row;
                        i++, l++) {
                    for (int j = this->corner_left.start_col, k = 0;
                            j <= this->corner_left.end_col;
                            j++, k++) {
                        this->data_source[l][k] = this->data_dest[l][k] = this->corner_left.buff[i][j];
                    }
                }
            }

            if (this->corner_right.buff != NULL) {
                // Copia corner right
                for (int i = this->corner_right.start_row, l = this->start_row;
                        i <= this->corner_right.end_row;
                        i++, l++) {
                    for (int j = this->corner_right.start_col, k = this->end_col + 1;
                            j <= this->corner_right.end_col;
                            j++, k++) {
                        this->data_source[l][k] = this->data_dest[l][k] = this->corner_right.buff[i][j];
                    }
                }
            }
        }
    } else {
        pthread_join(this->thread_id, NULL);
        this->num_iterators_ctrl++;
    }

    return;
}

/**
 * \brief Acessa as posições da Halo_Top, a fim de possibilitar o cálculo das
 *        bordas dos subdomínios.
 * 
 * As posições acessadas são sempre as últimas do array bidimensional que
 * armazena os dados da Halo_Top. Se houverem múltiplas Halos e for necessário o
 * cálculo desses pontos sobrepostos, é apenas a Halo_Top mais próxima da borda
 * do subdomínio que deverá ser utilizada pelo processo.       
 * 
 * @param [in] i  Índice.
 * @return Referência da posição do vetor.
 */
double& Halo_Top::operator[](int i) {
    return this->data_source[this->end_row][this->start_col + i];
}

/**
 * \brief Acessa as posições da Halo_Top em um contexto const.
 * 
 * As posições acessadas são sempre as últimas do array bidimensional que
 * armazena os dados da Halo_Top. Se houverem múltiplas Halos e for necessário o
 * cálculo desses pontos sobrepostos, é apenas a Halo_Top mais próxima da borda
 * do subdomínio que deverá ser utilizada pelo processo.       
 * 
 * @param [in] i  Índice.
 * @return Valor contido na posição do vetor.
 */
double Halo_Top::operator[](int i) const {
    return this->data_source[this->end_row][this->start_col + i];
}

/**
 * \brief Faz o dump dos dados da Halo_Top para o arquivo de nome filename.
 * 
 * @param [in] filename  Nome do arquivo.
 */
void Halo_Top::print_to_file(const char *filename) {
    if (this->rank_neighbor == MPI_PROC_NULL) {
        return;
    }

    fstream fout(filename, fstream::out | fstream::app);

    fout << "NotIter:" << endl;

    for (int i = 0; i < this->get_num_halos(); i++) {
        for (int j = 0; j < this->get_halo_size(); j++) {
            fout << this->data_source[i][j] << " ";
        }
        fout << endl;
    }

    fout.close();

    return;
}

/**
 * \brief Faz o dump dos dados da Halo_Top para o arquivo de nome filename.
 * 
 * @param [in] filename  Nome do arquivo.
 */
void Halo_Top::print_iter_to_file(const char *filename) {
    if (this->rank_neighbor == MPI_PROC_NULL) {
        return;
    }

    fstream fout(filename, fstream::out | fstream::app);

    fout << "Source:" << endl;

    for (int i = 0; i < this->get_num_halos(); i++) {
        for (halo_iterator hi = h_source[i]->first();; hi++) {
            fout << *hi << " ";
            if (hi == h_source[i]->last()) {
                break;
            }
        }
        fout << endl;
    }

    fout << "Dest:" << endl;

    for (int i = 0; i < this->get_num_halos(); i++) {
        for (halo_iterator hi = h_dest[i]->first();; hi++) {
            fout << *hi << " ";
            if (hi == h_dest[i]->last()) {
                break;
            }
        }
        fout << endl;
    }

    this->swap();

    fout << "Source:" << endl;

    for (int i = 0; i < this->get_num_halos(); i++) {
        for (halo_iterator hi = h_source[i]->first();; hi++) {
            fout << *hi << " ";
            if (hi == h_source[i]->last()) {
                break;
            }
        }
        fout << endl;
    }

    fout << "Dest:" << endl;

    for (int i = 0; i < this->get_num_halos(); i++) {
        for (halo_iterator hi = h_dest[i]->first();; hi++) {
            fout << *hi << " ";
            if (hi == h_dest[i]->last()) {
                break;
            }
        }
        fout << endl;
    }

    fout.close();

    return;
}

void Halo_Top::print_info(void) {
    this->logfile << "  this->num_rows = " << this->num_rows << endl;
    this->logfile << "  this->num_cols = " << this->num_cols << endl;

    this->logfile << "  this->start_row = " << this->start_row << endl;
    this->logfile << "  this->end_row = " << this->end_row << endl;
    this->logfile << "  this->start_col = " << this->start_col << endl;
    this->logfile << "  this->end_col = " << this->end_col << endl;

    this->logfile << "  Corner top:" << endl;
    this->logfile << "    this->corner_left.start_row = " << this->corner_left.start_row << endl;
    this->logfile << "    this->corner_left.end_row = " << this->corner_left.end_row << endl;
    this->logfile << "    this->corner_left.start_col = " << this->corner_left.start_col << endl;
    this->logfile << "    this->corner_left.end_col = " << this->corner_left.end_col << endl;

    this->logfile << "  Corner bottom:" << endl;
    this->logfile << "    this->corner_right.start_row = " << this->corner_right.start_row << endl;
    this->logfile << "    this->corner_right.end_row = " << this->corner_right.end_row << endl;
    this->logfile << "    this->corner_right.start_col = " << this->corner_right.start_col << endl;
    this->logfile << "    this->corner_right.end_col = " << this->corner_right.end_col << endl;
    this->logfile << "  this->halo_size = " << this->halo_size << endl;
    this->logfile << "  this->num_halos = " << this->num_halos << endl;
    this->logfile << "  this->num_iterators = " << this->num_iterators << endl;
    this->logfile << "  this->num_iterators_ctrl = " << this->num_iterators_ctrl << endl;

    this->logfile << "buff:" << endl;
    for (int i = 0; i < this->num_halos; i++) {
        for (int j = 0; j < this->halo_size; j++) {
            this->logfile << this->buff[i][j] << "  ";
        }
        this->logfile << endl;
    }
    this->logfile << "---------------------------------------" << endl << endl;

    return;
}

void Halo_Top::print_buff(double **buff, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            this->logfile << buff[i][j] << "  ";
        }
        this->logfile << endl;
    }

    return;
}
