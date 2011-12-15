# Makefile
makefile:
all: mesh2d

.PHONY: clean mesh2d

include Makefile.common

mesh2d:
	cd src ; make all

clean:
	cd obj ; make clean

