include ../../Makefile.common

.PHONY: topology

topology: ${OBJ_PATH}/topology/topology.o ${OBJ_PATH}/topology/topology_output.o

${OBJ_PATH}/topology/topology.o: topology.cpp
	${MPICPP} topology.cpp -o ${OBJ_PATH}/topology/topology.o -c -I ${MESH2D}/include ${CPP_FLAGS}

${OBJ_PATH}/topology/topology_output.o: topology_output.cpp
	${MPICPP} topology_output.cpp -o ${OBJ_PATH}/topology/topology_output.o -c -I ${MESH2D}/include ${CPP_FLAGS}

