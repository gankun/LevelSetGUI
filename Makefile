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

INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include -I/home/mcedeno/LevelSetGUI
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
# Please add any files you need to SOURCES below.
SOURCES = bab_gui.cpp
LIBS = -lGLEW -lGL -lGLU -lglut -lm

EXENAME = bab_gui

all: $(SOURCES)
	$(CC) $(FLAGS) $(EXENAME) $(INCLUDE) $(LIBDIR) $(SOURCES) $(LIBS)

clean:
	rm -f *.o $(EXENAME) 


.PHONY: all clean
