#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>

#define NBIN  10000000  // Number of bins
#define NUM_DEVICE 2    // Number of GPU devices to use
#define NUM_BLOCK   13  // Number of thread blocks
#define NUM_THREAD 192  // Number of threads per block

// Kernel that executes on the CUDA device
__global__ void cal_arctan(float *sum, int nbin, float step, float offset, int nthreads, int nblocks) {
    int i;
    float x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Sequential thread index across the blocks
    for (i = idx; i < nbin; i += nthreads * nblocks) {  // Interleaved bin assignment to threads
        x = offset + (i + 0.5) * step;
        sum[idx] += 1.0 / (1.0 + x * x);  // 1/(1+x*x) is being calculated
    }
}

int main(int argc, char **argv) {
    int myid, nproc, nbin, tid;
    float step, offset, arctan = 0.0, arctan_global;
    dim3 dimGrid(NUM_BLOCK, 1, 1);  // Grid dimensions (only use 1D)
    dim3 dimBlock(NUM_THREAD, 1, 1);  // Block dimensions (only use 1D)
    float *sumHost, *sumDev;  // Pointers to host & device arrays
    int dev_used;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // My MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);  // Number of MPI processes

    omp_set_num_threads(NUM_DEVICE); // One OpenMP thread per GPU device
    nbin = NBIN / (nproc * NUM_DEVICE); // # of bins per OpenMP thread
    step = 1.0 / (float)(nbin * nproc * NUM_DEVICE);

    #pragma omp parallel private(offset, sumHost, sumDev, tid, dev_used) reduction(+:arctan)
    {
        int mpid = omp_get_thread_num();
        offset = (NUM_DEVICE * myid + mpid) * step * nbin; // Quadrature-point offset
        cudaSetDevice(mpid % 2);

        size_t size = NUM_BLOCK * NUM_THREAD * sizeof(float);  // Array memory size
        sumHost = (float *)malloc(size);  // Allocate array on host
        cudaMalloc((void **)&sumDev, size);  // Allocate array on device
        cudaMemset(sumDev, 0, size);  // Reset array in device to 0

        // Calculate on device (call CUDA kernel)
        cal_arctan <<<dimGrid, dimBlock>>> (sumDev, nbin, step, offset, NUM_THREAD, NUM_BLOCK);

        // Retrieve result from device and store it in host array
        cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

        // Reduction over CUDA threads
        for (tid = 0; tid < NUM_THREAD * NUM_BLOCK; tid++)
            arctan += sumHost[tid];
        arctan *= step;

        // CUDA cleanup
        free(sumHost);
        cudaFree(sumDev);
        cudaGetDevice(&dev_used);
        printf("myid = %d; mpid = %d: device used = %d; partial arctan = %f\n", myid, mpid, dev_used, arctan);
    }

    // Reduction over MPI processes
    MPI_Allreduce(&arctan, &arctan_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (myid == 0) {
        printf("arctan(1) = %f\n", arctan_global);
    }

    MPI_Finalize();
    return 0;
}