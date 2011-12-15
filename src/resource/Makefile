include ../../Makefile.common

.PHONY: resource

resource: ${OBJ_PATH}/resource/resource.o

${OBJ_PATH}/resource/resource.o: resource.cpp
	${MPICPP} resource.cpp -o ${OBJ_PATH}/resource/resource.o -c -I ${MESH2D}/include ${CPP_FLAGS}

