CC := g++
CXXFLAGS := -o3 -Wall -std=c++14 -Wdeprecated-declarations

INCLUDE := -I./include -I./include/dgsparse
LIBRARY := -ldl

#cuda 11.0 wrapper
INCLUDE_11_0 := -I./include/cuda-11.0
SRC_CUDA_11_0 := $(wildcard ./src/*.cc) $(wildcard ./src/cuda-11.0/*.cc)
OBJ_CUDA_11_0 := $(patsubst %.c %.cc %.cpp,%.o, $(SRC_CUDA_11_0))
OUTPUT_NAME_CUDA_11_0 := libdgsparsewrapper.so.11.0

#cuda 11.1 wrapper
INCLUDE_11_1 := -I./include/cuda-11.1
SRC_CUDA_11_1 := $(wildcard ./src/*.cc) $(wildcard ./src/cuda-11.1/*.cc)
OBJ_CUDA_11_1 := $(patsubst %.c %.cc %.cpp,%.o, $(SRC_CUDA_11_1))
OUTPUT_NAME_CUDA_11_1 := libdgsparsewrapper.so.11.1

#cuda 11.2 wrapper
INCLUDE_11_2 := -I./include/cuda-11.2
SRC_CUDA_11_2 := $(wildcard ./src/*.cc) $(wildcard ./src/cuda-11.2/*.cc)
OBJ_CUDA_11_2 := $(patsubst %.c %.cc %.cpp,%.o, $(SRC_CUDA_11_2))
OUTPUT_NAME_CUDA_11_2 := libdgsparsewrapper.so.11.2

#cuda 11.3 wrapper
INCLUDE_11_3 := -I./include/cuda-11.3
SRC_CUDA_11_3 := $(wildcard ./src/*.cc) $(wildcard ./src/cuda-11.3/*.cc)
OBJ_CUDA_11_3 := $(patsubst %.c %.cc %.cpp,%.o, $(SRC_CUDA_11_3))
OUTPUT_NAME_CUDA_11_3 := libdgsparsewrapper.so.11.3

OUTPUT_PATH := ./bin

.PHONY : clean

all : $(OUTPUT_NAME_CUDA_11_0) $(OUTPUT_NAME_CUDA_11_1) $(OUTPUT_NAME_CUDA_11_2) $(OUTPUT_NAME_CUDA_11_3) 

clean :
	rm -f $(OUTPUT_PATH)/*

$(OUTPUT_NAME_CUDA_11_0): $(OBJ_CUDA_11_0)
	$(CC) $(CXXFLAGS) $(INCLUDE) $(INCLUDE_11_0) -DDISABLE_CUSPARSE_DEPRECATED -fPIC -shared -o $@ $(OBJ_CUDA_11_0) $(LIBRARY)
	mkdir -p $(OUTPUT_PATH)
	mv $@ $(OUTPUT_PATH)	

$(OUTPUT_NAME_CUDA_11_1): $(OBJ_CUDA_11_1)
	$(CC) $(CXXFLAGS) $(INCLUDE) $(INCLUDE_11_1) -DDISABLE_CUSPARSE_DEPRECATED -fPIC -shared -o $@ $(OBJ_CUDA_11_1) $(LIBRARY)
	mkdir -p $(OUTPUT_PATH)
	mv $@ $(OUTPUT_PATH)

$(OUTPUT_NAME_CUDA_11_2): $(OBJ_CUDA_11_2)
	$(CC) $(CXXFLAGS) $(INCLUDE) $(INCLUDE_11_2) -DDISABLE_CUSPARSE_DEPRECATED -fPIC -shared -o $@ $(OBJ_CUDA_11_2) $(LIBRARY)
	mkdir -p $(OUTPUT_PATH)
	mv $@ $(OUTPUT_PATH)

$(OUTPUT_NAME_CUDA_11_3): $(OBJ_CUDA_11_3)
	$(CC) $(CXXFLAGS) $(INCLUDE) $(INCLUDE_11_3) -DDISABLE_CUSPARSE_DEPRECATED -fPIC -shared -o $@ $(OBJ_CUDA_11_3) $(LIBRARY)
	mkdir -p $(OUTPUT_PATH)
	mv $@ $(OUTPUT_PATH)