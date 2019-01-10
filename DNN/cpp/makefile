.PHONY : all clean
CXX=g++

all : engine test

engine : dnn_engine_pybind.o dnn_engine.o
	$(CXX) -shared -fopenmp dnn_engine_pybind.o dnn_engine.o -o dnn_engine_pybind.so

test :
	$(CXX) -I/home/mji/.local/include dnn_engine_test.cpp dnn_engine.cpp -pthread -lgtest -o dnn_engine_test

dnn_engine.o : dnn_engine.cpp
	$(CXX) -c -std=c++14 -march=native -fPIC -O3 -fopenmp -I/home/mji/.local/include $^

dnn_engine_pybind.o : dnn_engine_pybind.cpp dnn_engine.h
	$(CXX) -c -std=c++14 -march=native -fPIC -O3 -fopenmp -I/home/mji/.local/include $(shell python3 -m pybind11 --includes) $<

dnn_engine_test.o : dnn_engine_test.cpp dnn_engine.h
	$(CXX) -c -std=c++14 -I/home/mji/.local/include $<

clean :
	rm -f *.o dnn_engine_pybind.so dnn_engine_test
