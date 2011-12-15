include ../../Makefile.common

.PHONY: mesh2d

mesh2d: ${OBJ_PATH}/mesh2d/mesh2d.o ${OBJ_PATH}/mesh2d/ghost_zones/ghost_zones.o ${OBJ_PATH}/mesh2d/ghost_zones/halo_left.o ${OBJ_PATH}/mesh2d/ghost_zones/halo_right.o ${OBJ_PATH}/mesh2d/ghost_zones/halo_top.o ${OBJ_PATH}/mesh2d/ghost_zones/halo_bottom.o

${OBJ_PATH}/mesh2d/mesh2d.o: mesh2d.cpp
	${MPICPP} mesh2d.cpp -o ${OBJ_PATH}/mesh2d/mesh2d.o -c -I ${MESH2D}/include ${CPP_FLAGS}

${OBJ_PATH}/mesh2d/ghost_zones/ghost_zones.o: ghost_zones/ghost_zones.cpp ghost_zones/halo_left.cpp ghost_zones/halo_right.cpp ghost_zones/halo_top.cpp ghost_zones/halo_bottom.cpp
	cd ghost_zones ; make all

