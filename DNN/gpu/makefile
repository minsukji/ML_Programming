.PHONY : all clean
CXX=nvcc

all : engine test

engine : dnn_engine_gpu_pybind.o dnn_engine_gpu.o
	$(CXX) -shared dnn_engine_gpu_pybind.o dnn_engine_gpu.o -lcublas -lcurand -o dnn_engine_pybind.so

dnn_engine_gpu.o : dnn_engine_gpu.cu
	$(CXX) --device-c -std=c++14 -Xcompiler -fPIC,-Wall $^

dnn_engine_gpu_pybind.o : dnn_engine_gpu_pybind.cu dnn_engine_gpu.h
	$(CXX) --device-c -std=c++14 -Xcompiler -fPIC,-Wall $(shell python3 -m pybind11 --includes) $<

test :
	$(CXX) -std=c++14 dnn_engine_gpu_test.cu dnn_engine_gpu.cu -lcublas -lcurand -o dnn_engine_gpu_test

clean :
	rm -f *.o dnn_engine_pybind.so dnn_engine_gpu_test
