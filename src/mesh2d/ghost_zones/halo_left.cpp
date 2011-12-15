#include <halo.h>

void* jacobi2d_left(void *args) {
    Halo_Left *This = (Halo_Left*) args;

    for (int halo = This->num_iterators_ctrl; halo < This->num_iterators; halo++) {
        for (halo_iterator hs = This->h_source[halo]->first(),
                hd = This->h_dest[halo]->first();; hs++, hd++) {
            *hd = hs.reduce() * 0.25;
            if (hs == This->h_source[halo]->last()) {
                break;
            }
        }
    }

    halo_iterator hd = This->h_dest[This->num_iterators - 1]->first();

    for (int i = 0; i < This->mesh.size_y; i++, hd++) {
        *hd = *hd + ((*(This->mesh.mesh))[i][0] * 0.25);
    }

    pthread_exit(NULL);
}

/**
 * \brief Construtor do Halo_Left, sobrepondo-o com o vizinho de rank rank_neighbor.
 * 
 * @param [in] halo_size      Tamanho do Halo_Left.
 * @param [in] num_halos      Número de Halos.
 * @param [in] rank_neighbor  Rank do vizinho com o qual ele sobrepõe-se.
 * @param [in] direction_tag  Tag utilizado para receber a borda do vizinho.
 * @param [in] comm           Comunicador MPI.
 * @return
 */
Halo_Left::Halo_Left(int halo_size, int num_halos, struct mesh_t mesh,
        struct diag_buff_t *corner_top, struct diag_buff_t *corner_bottom,
        int rank_neighbor, int direction_tag, MPI_Comm comm) {

    this->num_rows = num_halos * HALO_THICKNESS + (2 * HALO_THICKNESS);
    this->num_cols = halo_size + (2 * HALO_THICKNESS);

    this->start_row = HALO_THICKNESS;
    this->end_row = this->num_rows - HALO_THICKNESS - 1;
    this->start_col = HALO_THICKNESS;
    this->end_col = this->num_cols - HALO_THICKNESS - 1;

    this->mesh = mesh;

    // Inicializa as informações do canto superior e inferior
    //
    // Referência ao canto SUPERIOR. O espaço é alocado em Ghost_Zones
    this->corner_top.buff = corner_top->buff;
    // Definição dos índices específicos ao canto superior
    //   Índice da linha onde inicia a parte pertinente à região de sobreposição
    //   A linha começa a partir do final do canto enviado, menos a espessura
    //   do estêncil
    this->corner_top.start_row = corner_top->nrows - HALO_THICKNESS;
    //   Índice da linha onde termina a parte pertinente à região de sobreposição
    this->corner_top.end_row = corner_top->nrows - 1;
    //   Índice da coluna onde inicia a parte pertinente à região de sobreposição
    this->corner_top.start_col = 0;
    //   Índice da coluna onde termina a parte pertinente à região de sobreposição
    this->corner_top.end_col = corner_top->ncols - 1;
    // Número de linhas
    this->corner_top.nrows = corner_top->nrows;
    // Número de colunas
    this->corner_top.ncols = corner_top->ncols;
    // Rank do processo diagonal que enviou o canto. Isso é usado apenas para debug
    this->corner_top.rank_source = corner_top->rank_source;

    // Referência ao canto INFERIOR. O espaço é alocado em Ghost_Zones
    this->corner_bottom.buff = corner_bottom->buff;
    // Definição dos índices específicos ao canto superior
    //   Índice da linha onde inicia a parte pertinente à região de sobreposição
    this->corner_bottom.start_row = 0;
    this->corner_bottom.end_row = HALO_THICKNESS - 1;
    this->corner_bottom.start_col = 0;
    this->corner_bottom.end_col = corner_bottom->ncols - 1;
    this->corner_bottom.nrows = corner_bottom->nrows;
    this->corner_bottom.ncols = corner_bottom->ncols;
    this->corner_bottom.rank_source = corner_bottom->rank_source;

    pthread_attr_init(&this->thread_attr);
    pthread_attr_setdetachstate(&this->thread_attr, PTHREAD_CREATE_JOINABLE);

    // Tamanho do Halo_Left.
    this->halo_size = halo_size;

    // Número de Halos.
    this->num_halos = num_halos;

    // O vizinho é nulo. Pode ser alterado mais tarde por meio da chamada
    // do mesmo método.
    this->set_neighbor_info(rank_neighbor, direction_tag, comm);

    this->data_u = this->data_v = this->data_source = this->data_dest = NULL;
    this->buff = NULL;
    this->h_source = this->h_dest = NULL;

    this->print_info();

    // Aloca o Halo_Left.
    this->alloc();
    // Define os iteradores.
    this->set_iterators();


}

/**
 * \brief Destrutor.
 * @return
 */
Halo_Left::~Halo_Left() {
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
 * \brief Aloca o vetor bidimensional que conterá os dados do Halo_Left.
 * @return Primeira posição de memória do espaço alocado.
 */
void Halo_Left::alloc(void) {

    alloc_cont_array2d(&(this->data_u), this->num_rows, this->num_cols);
    // Se tiver mais de uma região de sobreposição,
    // aloca o array2d adicional para possibilitar o cálculo
    if (this->num_halos > 1) {
        alloc_cont_array2d(&(this->data_v), this->num_rows, this->num_cols);
    }

    this->data_source = this->data_u;
    this->data_dest = this->data_v;

    alloc_cont_array2d(&(this->buff), this->get_num_halos(), this->get_halo_size());

    return;
}

void Halo_Left::set_iterators(void) {
    if (this->get_num_halos() <= 1) {
        this->num_iterators = this->num_iterators_ctrl = 0;
        this->h_source = this->h_dest = NULL;
        return;
    }

    // Os iteradores devem desconsiderar as linhas e colunas extras do halo,
    // utilizadas apenas para manter as referências do stencil válidas.
    this->num_iterators = this->num_rows - (2 * HALO_THICKNESS);
    // Inicializa num_iterators_ctrl para que a primeira vez tranfira as bordas
    this->num_iterators_ctrl = this->num_iterators;

    this->h_source = new H*[this->num_iterators];
    this->h_dest = new H*[this->num_iterators];

    for (int i = 0, j = this->start_row; i < this->num_iterators; i++, j++) {
        this->h_source[i] = new H(&(this->data_source), j, this->start_col, this->get_halo_size(), this->num_cols);
    }

    for (int i = 0, j = this->start_row; i < this->num_iterators; i++, j++) {
        this->h_dest[i] = new H(&(this->data_dest), j, this->start_col, this->get_halo_size(), this->num_cols);
    }

    return;
}

void Halo_Left::swap(void) {
    double **temp = this->data_source;

    this->data_source = this->data_dest;
    this->data_dest = temp;

    return;
}

/**
 * \brief Sobrepõe o Halo_Left com um vizinho.
 * 
 * @param [in] rank_neighbor  Rank do vizinho.
 * @param [in] direction_tag  Tag da mensagem.
 * @param [in] comm           Comunicador MPI.
 */
void Halo_Left::set_neighbor_info(int rank_neighbor, int direction_tag, MPI_Comm comm) {
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
int Halo_Left::get_neighbor_rank(void) {
    return this->rank_neighbor;
}

/**
 * \brief Devolve o tamanho do Halo_Left.
 * @return Tamanho do Halo_Left.
 */
int Halo_Left::get_halo_size(void) {
    return this->halo_size;
}

/**
 * \brief Devolve o número de Halos.
 * @return Número de Halos.
 */
int Halo_Left::get_num_halos(void) {
    return this->num_halos;
}

/**
 * \brief Inicia o recebimento assíncrono da borda do vizinho.
 * 
 * O recebimento é feito por meio da primitiva MPI_Irecv, que espera uma
 * mensagem do vizinho informado ou no momento da instanciação do Halo_Left
 * ou no momento da chamada ao método set_neighbor_info. A mensagem é aguardada
 * com o tag e o comunicador passados por parâmetros.
 */
bool Halo_Left::init_update(void) {
    // Não recebe de Halos não sobrepostos.
    if (this->rank_neighbor == MPI_PROC_NULL) {
        return true;
    }

    if (this->transfer()) {
        MPI_Irecv(this->buff[0], // buff
                this->num_halos * this->halo_size * sizeof (double), // count
                MPI_PACKED, // datatype
                this->rank_neighbor, // source
                this->direction_tag, // tag
                this->comm, // comm
                &(this->request)); // request

        return true;
    } else {
        pthread_create(&(this->thread_id), &(this->thread_attr), jacobi2d_left, this);
        return false;
    }

    return true;
}

bool Halo_Left::transfer(void) {
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
void Halo_Left::sync(void) {
    // Ignora Halos não sobrepostos.
    if (this->rank_neighbor == MPI_PROC_NULL) {
        return;
    }

    int pos = 0;

    if (this->transfer()) {
        // Simula latência de rede
        //usleep(500);
        //////////////////////////
        MPI_Wait(&(this->request), MPI_STATUS_IGNORE);

        for (int i = this->start_row; i <= this->end_row; i++) {
            MPI_Unpack(this->buff[0], // inbuf
                    this->num_halos * this->halo_size * sizeof (double), // insize
                    &pos, // position
                    this->data_source[i] + HALO_THICKNESS, // outbuff
                    this->halo_size, // outcount (number of items to be unpacked)
                    MPI_DOUBLE, // datatype
                    this->comm); // comm
        }

        if (this->num_halos > 1) {
            memcpy(this->data_dest[0], this->data_source[0],
                    this->num_rows * this->num_cols * sizeof (double));

            this->num_iterators_ctrl = 0;

            if (this->corner_top.buff != NULL) {
                // Copia corner top
                for (int i = this->corner_top.start_row, k = 0;
                        i <= this->corner_top.end_row;
                        i++, k++) {
                    for (int j = this->corner_top.start_col, l = this->start_row;
                            j <= this->corner_top.end_col;
                            j++, l++) {
                        this->data_source[l][k] = this->data_dest[l][k] = corner_top.buff[i][j];
                    }
                }
            }
            

            // Copia corner bottom
            if (this->corner_bottom.buff != NULL) {
                for (int i = this->corner_bottom.start_row, k = this->end_col + 1;
                        i <= this->corner_bottom.end_row;
                        i++, k++) {
                    for (int j = this->corner_bottom.start_col, l = this->start_row;
                            j <= this->corner_bottom.end_col;
                            j++, l++) {
                        this->data_source[l][k] = this->data_dest[l][k] = corner_bottom.buff[i][j];
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
 * \brief Acessa as posições da Halo_Left, a fim de possibilitar o cálculo das
 *        bordas dos subdomínios.
 * 
 * As posições acessadas são sempre as últimas do array bidimensional que
 * armazena os dados da Halo_Left. Se houverem múltiplas Halos e for necessário o
 * cálculo desses pontos sobrepostos, é apenas a Halo_Left mais próxima da borda
 * do subdomínio que deverá ser utilizada pelo processo.       
 * 
 * @param [in] i  Índice.
 * @return Referência da posição do vetor.
 */
double& Halo_Left::operator[](int i) {
    return this->data_source[this->end_row][this->start_col + i];
}

/**
 * \brief Acessa as posições da Halo_Left em um contexto const.
 * 
 * As posições acessadas são sempre as últimas do array bidimensional que
 * armazena os dados da Halo_Left. Se houverem múltiplas Halos e for necessário o
 * cálculo desses pontos sobrepostos, é apenas a Halo_Left mais próxima da borda
 * do subdomínio que deverá ser utilizada pelo processo.       
 * 
 * @param [in] i  Índice.
 * @return Valor contido na posição do vetor.
 */
double Halo_Left::operator[](int i) const {
    return this->data_source[this->end_row][this->start_col + i];
}

/**
 * \brief Faz o dump dos dados da Halo_Left para o arquivo de nome filename.
 * 
 * @param [in] filename  Nome do arquivo.
 */
void Halo_Left::print_to_file(const char *filename) {
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
 * \brief Faz o dump dos dados da Halo_Left para o arquivo de nome filename.
 * 
 * @param [in] filename  Nome do arquivo.
 */
void Halo_Left::print_iter_to_file(const char *filename) {
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

void Halo_Left::print_info(void) {


    this->logfile << "  this->num_rows = " << this->num_rows << endl;
    this->logfile << "  this->num_cols = " << this->num_cols << endl;

    this->logfile << "  this->start_row = " << this->start_row << endl;
    this->logfile << "  this->end_row = " << this->end_row << endl;
    this->logfile << "  this->start_col = " << this->start_col << endl;
    this->logfile << "  this->end_col = " << this->end_col << endl;

    this->logfile << "  Corner top:" << endl;
    this->logfile << "    this->corner_top.start_row = " << this->corner_top.start_row << endl;
    this->logfile << "    this->corner_top.end_row = " << this->corner_top.end_row << endl;
    this->logfile << "    this->corner_top.start_col = " << this->corner_top.start_col << endl;
    this->logfile << "    this->corner_top.end_col = " << this->corner_top.end_col << endl;

    this->logfile << "  Corner bottom:" << endl;
    this->logfile << "    this->corner_bottom.start_row = " << this->corner_bottom.start_row << endl;
    this->logfile << "    this->corner_bottom.end_row = " << this->corner_bottom.end_row << endl;
    this->logfile << "    this->corner_bottom.start_col = " << this->corner_bottom.start_col << endl;
    this->logfile << "    this->corner_bottom.end_col = " << this->corner_bottom.end_col << endl;
    this->logfile << "  this->halo_size = " << this->halo_size << endl;
    this->logfile << "  this->num_halos = " << this->num_halos << endl;
    this->logfile << "  this->num_iterators = " << this->num_iterators << endl;
    this->logfile << "  this->num_iterators_ctrl = " << this->num_iterators_ctrl << endl;
    this->logfile << "---------------------------------------" << endl << endl;

    return;
}

void Halo_Left::print_buff(double **buff, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            this->logfile << buff[i][j] << "  ";
        }
        this->logfile << endl;
    }

    return;
}
