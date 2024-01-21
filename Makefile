CUDA_PATH     ?= C:\programming\cuda\13.2
HOST_COMPILER  = cl
NVCC           = $(CUDA_PATH)\bin\nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
# NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_61,code=sm_61

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h

cudart.exe: cudart.obj
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.exe cudart.obj

cudart.obj: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.obj -c main.cu

out.ppm: cudart.exe
	del /Q out.ppm
	cudart.exe > out.ppm

out.jpg: out.ppm
	del /Q out.jpg
	cmd /c magick out.ppm out.jpg

profile_basic: cudart.exe
	nvprof cudart.exe > out.ppm

# use nvprof --query-metrics
profile_metrics: cudart.exe
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer cudart.exe > out.ppm

clean:
	del /Q cudart.exe cudart.obj out.ppm out.jpg
