gpu : gpu_engine.o activation_func.o adam.o backward.o compute_cost.o dropout.o forward.o \
 initialize.o momentum.o setup.o update.o
	nvcc -shared -Xcompiler -fPIC,-Wall $^ -lcublas -lcurand -o gpu_engine.so

gpu_engine.o : gpu_engine.cu activation_func.h adam.h backward.h compute_cost.h dropout.h forward.h \
 initialize.h momentum.h setup.h update.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $(shell python3 -m pybind11 --includes) $<

activation_func.o : activation_func.cu activation_func.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

adam.o : adam.cu adam.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

backward.o : backward.cu activation_func.h backward.h dropout.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

compute_cost.o : compute_cost.cu compute_cost.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

dropout.o : dropout.cu dropout.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

forward.o : forward.cu activation_func.h dropout.h forward.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

initialize.o : initialize.cu initialize.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

momentum.o : momentum.cu momentum.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

setup.o : setup.cu setup.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

update.o : update.cu update.h
	nvcc --device-c -std=c++14 -Xcompiler -fPIC,-Wall $<

