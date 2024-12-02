module purge
module load usc/8.3.0
module load cuda

nvcc -Xcompiler -fopenmp dist_comp.cu -o dist_comp -I${OPENMPI_ROOT}/include-L${OPENMPI_ROOT}/lib -lmpi -lgomp