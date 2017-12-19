
#INCLUDE = /usr/local/cuda-6.5/include 
CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include
#HEADER=/home/arashh/GPU/labs/lab2/sobel_handout/helper_image.h
HEADER=-I/usr/local/cuda/samples/common/inc
#CC = /usr/local/cuda-6.5/bin/nvcc
#CC = g++
#CC = nvcc
FLAGS = -g -Wall
SOURCES = sobel.cu
#SOURCES = sobel.c

EXECUTABLES = sobel

all:
	$(CC) $(HEADER) $(CFLAS) $(SOURCES) -o $(EXECUTABLES)
	./$(EXECUTABLES)
clean:
        
	rm -rf sobel.o
