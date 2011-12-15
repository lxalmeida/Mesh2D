#ifndef STENCIL_H
#define STENCIL_H

// Classe Stencil abstrata: "Um stencil pode ser qualquer coisa"
class Stencil {
    public:
        virtual void *compute(void *args) = 0; // função virtual pura
};

#endif

