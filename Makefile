CC := g++
CXXFLAGS := -o3 -Wall -std=c++14 -Wdeprecated-declarations

INCLUDE := -I./include -I./include/cuda-11.1 -I./include/dgsparse-0.1
LIBRARY := -ldl

SRC_CUDA_11_1 := $(wildcard ./src/*.cc) $(wildcard ./src/cuda-11.1/*.cc)
OBJ_CUDA_11_1 := $(patsubst %.c %.cc %.cpp,%.o, $(SRC_CUDA_11_1))
OUTPUT_NAME_CUDA_11_1 := libdgsparsewrapper.so.11.1

OUTPUT_PATH := ./bin

.PHONY : clean

all : $(OUTPUT_NAME_CUDA_11_1)

clean :
	rm -f $(OUTPUT_PATH)/*

$(OUTPUT_NAME_CUDA_11_1): $(OBJ_CUDA_11_1OBJ_CUDA_11_1)
	$(CC) $(CXXFLAGS) $(INCLUDE) -DDISABLE_CUSPARSE_DEPRECATED -fPIC -shared -o $@ $(OBJ_CUDA_11_1) $(LIBRARY)
	mkdir -p $(OUTPUT_PATH)
	mv $@ $(OUTPUT_PATH)
