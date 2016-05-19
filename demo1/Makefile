###############################################################################
# CS/CNS 171 Fall 2014
#
# This is a template Makefile for Assignment 3. It is meant to compile your
# OpenGL program while also taking into account the Flex and Bison dependence.
# Edit it however you find convenient.
# 
# The current version of this file should compile your program just fine on
# Debian-based Linux operating systems.
#
# If you run Mac OS or other distributions of Linux, then you may have to
# fool around with the 'INCLUDE' and 'LIBDIR' lines below before the Makefile
# can compile the OpenGL parts successfully.
###############################################################################
CC = g++
FLAGS = -g -o

CUDA_PATH ?= /usr/local/cuda-7.5
CUDA_INC_PATH ?= $(CUDA_PATH)/include
CUDA_BIN_PATH ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib

NVCC          ?= $(CUDA_BIN_PATH)/nvcc


# Modify the -I/home/mcedeno/... line to the location of the LevelSetsGUI directory
INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include -I/home/mcedeno/projects/LevelSetGUI
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
LIBS = -lGLEW -lGL -lGLU -lglut -lm

EXENAME = stream_gui

all: $(EXENAME)


#stream_gui: ../bab_gui.cpp stream_gui.o 
#	$(CC) -std=c++11  $^ -o $@ -O3  $(INCLUDE) $(LIBDIR) $(LIBS) -Wall -I$(CUDA_INC_PATH)

#stream_gui.o: bab_stream.cu
#	$(NVCC) $<  -O3 -g -std=c++11 --expt-relaxed-constexpr-I$(CUDA_INC_PATH) -o $@ $(INCLUDE)


stream_gui: bab_stream.cu gui.o
	$(NVCC) $< -g -std=c++11 --expt-relaxed-constexpr-I$(CUDA_INC_PATH)

gui.o: ../bab_gui.cpp
	$(CC) -std=c++11 $^ $(INCLUDE) $(LIBDIR) $(LIBS)

clean:
	rm -f *.o $(EXENAME) 


.PHONY: all clean
