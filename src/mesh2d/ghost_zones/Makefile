include ../../../Makefile.common



.PHONY: all ghost_zones halo_left halo_right halo_top halo_bottom

all: ghost_zones halo_left halo_right halo_top halo_bottom

ghost_zones: ${OBJ_PATH}/mesh2d/ghost_zones/ghost_zones.o
halo_left: ${OBJ_PATH}/mesh2d/ghost_zones/halo_left.o
halo_right: ${OBJ_PATH}/mesh2d/ghost_zones/halo_right.o
halo_top: ${OBJ_PATH}/mesh2d/ghost_zones/halo_top.o
halo_bottom: ${OBJ_PATH}/mesh2d/ghost_zones/halo_bottom.o

${OBJ_PATH}/mesh2d/ghost_zones/ghost_zones.o: ghost_zones.cpp
	${MPICPP} ghost_zones.cpp -c -I ${MESH2D}/include -o ${OBJ_PATH}/mesh2d/ghost_zones/ghost_zones.o ${CPP_FLAGS}

${OBJ_PATH}/mesh2d/ghost_zones/halo_left.o: halo_left.cpp
	${MPICPP} halo_left.cpp -c -I ${MESH2D}/include -o ${OBJ_PATH}/mesh2d/ghost_zones/halo_left.o ${CPP_FLAGS}

${OBJ_PATH}/mesh2d/ghost_zones/halo_right.o: halo_right.cpp
	${MPICPP} halo_right.cpp -c -I ${MESH2D}/include -o ${OBJ_PATH}/mesh2d/ghost_zones/halo_right.o ${CPP_FLAGS}

${OBJ_PATH}/mesh2d/ghost_zones/halo_top.o: halo_top.cpp
	${MPICPP} halo_top.cpp -c -I ${MESH2D}/include -o ${OBJ_PATH}/mesh2d/ghost_zones/halo_top.o ${CPP_FLAGS}

${OBJ_PATH}/mesh2d/ghost_zones/halo_bottom.o: halo_bottom.cpp
	${MPICPP} halo_bottom.cpp -c -I ${MESH2D}/include -o ${OBJ_PATH}/mesh2d/ghost_zones/halo_bottom.o ${CPP_FLAGS}


