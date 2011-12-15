makefile:
all: mesh2d

include ../Makefile.common

.PHONY: mesh2d

mesh2d: ${OBJ_PATH}/topology/topology.o ${OBJ_PATH}/mesh2d/mesh2d.o ${OBJ_PATH}/mesh2d/ghost_zones/ghost_zones.o ${OBJ_PATH}/mesh2d/ghost_zones/halo_left.o  ${OBJ_PATH}/mesh2d/ghost_zones/halo_right.o  ${OBJ_PATH}/mesh2d/ghost_zones/halo_top.o  ${OBJ_PATH}/mesh2d/ghost_zones/halo_bottom.o ${OBJ_PATH}/resource/resource.o

${OBJ_PATH}/topology/topology.o: topology/topology.cpp
	cd topology ; make topology

${OBJ_PATH}/mesh2d/mesh2d.o: mesh2d/mesh2d.cpp mesh2d/ghost_zones/ghost_zones.cpp mesh2d/ghost_zones/halo_left.cpp mesh2d/ghost_zones/halo_right.cpp mesh2d/ghost_zones/halo_top.cpp mesh2d/ghost_zones/halo_bottom.cpp
	cd mesh2d ; make mesh2d

${OBJ_PATH}/resource/resource.o: resource/resource.cpp
	cd resource ; make resource

