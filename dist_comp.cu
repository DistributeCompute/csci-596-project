// DistributeCompute: Parallel Computation using MPI and CUDA
// Apery's Constant

// Hybrid MPI+CUDA computation of Pi
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <math.h>

#define NUM_DEVICE 2 // # of GPU devices = # of OpenMP threads
#define NBIN  100000000  // Number of bins, increasing for accuracy

__global__ void cal_apery(double *sum, int thread_id_global, int nthreads_total, int nterms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global index 
    if (idx < nterms) {
        // Compute the actual term index n for this thread
        int n = thread_id_global + 1 + idx * nthreads_total;
        if (n <= NBIN) {
            double dn = (double)n;
            sum[idx] = 1.0/(dn*dn*dn);
        } else {
            sum[idx] = 0.0;
        }
    }
}


int main(int argc, char **argv) {
    int myid, nproc;
    double apery = 0.0, apery_final;
    double *sumHost, *sumDev;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); // Number of MPI processes

    // Set number of threads = number of GPUs per MPI rank
    omp_set_num_threads(NUM_DEVICE);

    int nthreads_total = nproc * NUM_DEVICE; // Total number of (MPI×OMP) "threads"

    #pragma omp parallel private(sumHost, sumDev) reduction(+:apery)
    {
        int mpid = omp_get_thread_num(); // OMP thread id
        int thread_id_global = myid * NUM_DEVICE + mpid; // Global thread ID across all ranks and threads

        // Assign GPU device
        cudaSetDevice(mpid % NUM_DEVICE);

        // Number of terms this particular thread will handle
        int nterms = (NBIN - thread_id_global + nthreads_total - 1) / nthreads_total;
        if (nterms < 0) nterms = 0; // Safety check

        size_t size = nterms * sizeof(double);
        sumHost = (double *)malloc(size);
        cudaMalloc((void **)&sumDev, size);

        int threadsPerBlock = 256;
        int blocksPerGrid = (nterms + threadsPerBlock - 1) / threadsPerBlock;

        // Launch CUDA kernel
        cal_apery<<<blocksPerGrid, threadsPerBlock>>>(sumDev, thread_id_global, nthreads_total, nterms);

        // Copy results back to host
        cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

        // Sum partial results on host
        double partial_sum = 0.0;
        for (int i = 0; i < nterms; i++)
            partial_sum += sumHost[i];

        apery += partial_sum;

        int dev_used;
        cudaGetDevice(&dev_used);

        // Debug print
        printf("myid = %d; mpid = %d; device = %d; partial zeta(3) sum = %.15f\n", myid, mpid, dev_used, partial_sum);

        // Cleanup
        free(sumHost);
        cudaFree(sumDev);
    }

    // MPI reduce to get global sum
    MPI_Allreduce(&apery, &apery_final, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("Apery’s constant apery(3) ≈ %.30f\n", apery_final);
    }

    MPI_Finalize();
    return 0;
}
