CC := g++
CXXFLAGS := -o3 -Wall -std=c++14

INCLUDE := -I./include -I./include/cuda-11.1 -I./include/dgsparse-0.1
LIBRARY := -ldl

SRC := $(wildcard ./src/*.cc)
OBJ := $(patsubst %.c %.cc %.cpp,%.o, $(SRC))

OUTPUT_NAME := libdgsparsewrapper.so
OUTPUT_PATH := ./bin

.PHONY : clean

all : $(OUTPUT_NAME)

clean :
	rm -f $(OUTPUT_PATH)/*

$(OUTPUT_NAME): $(OBJ)
	$(CC) $(CXXFLAGS) $(INCLUDE) -fPIC -shared -o $@ $(OBJ) $(LIBRARY)
	mkdir -p $(OUTPUT_PATH)
	mv $@ $(OUTPUT_PATH)
