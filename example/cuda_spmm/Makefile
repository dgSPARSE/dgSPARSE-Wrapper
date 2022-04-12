all: spmm.out


INC = -I$(CUDA_HOME)/include
LOADPATH = -L$(CUDA_HOME)/lib64

spmm.out: spmm.o $(LIB)
	nvcc  $(INC) $(LOADPATH) -lcusparse -o $@ $^

%.o: %.cu
	nvcc -std=c++11 $(INC) -c $< 

clean:
	rm -f *.out *.o