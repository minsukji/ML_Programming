CXX=g++
IDIR=-I/home/mji/.local/include $(shell python3 -m pybind11 --includes) -I.
DEPS=dnn_engine.h
CXXFLAGS=-c -std=c++14 -Wall -march=native -fPIC -O3 -fopenmp $(IDIR)
LDFLAGS=-shared -fopenmp
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@

dnn_engine_pybind: $(OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@.so

.PHONY: clean

clean:
	rm -f $(OBJ)
