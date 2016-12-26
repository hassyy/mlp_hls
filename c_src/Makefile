CXX=   g++
PROG= mlp_exe
LIBS= -lm
SRC=	mlp_tb.cpp \
		mlp_top.cpp
HEAD=	mlp.h \
		mlp_top.h \
		mlp_config.h

VIVADO_HLS_INC_DIR="/mnt/hgfs/vm_share/vivado_hls/include/"

INC_DIR += -I$(VIVADO_HLS_INC_DIR)

OBJ= $(SRC:.c=.o)

$(PROG): $(OBJ) $(HEAD)
	$(CXX) $(CXXFLAG) $(LIBS) $(OBJ) -o $(PROG) $(INC_DIR)

$(OBJ): $(HEAD)

.c.o:
	$(CXX) $(INCS) -c $<

clean:
	$(RM) -f *.o
	
