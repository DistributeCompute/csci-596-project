// Hybrid MPI+CUDA computation of the Catalan’s constant G
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <math.h>

#define N 1000000   // Number of terms in the series (adjustable for accuracy)
#define NUM_DEVICE 2 // Number of GPU devices = Number of OpenMP threads

// CUDA Kernel to compute partial sums for Catalan's constant
__global__ void cal_G(double *sum, int thread_id_global, int nthreads_total, int nterms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index within this device/thread group
    if (idx < nterms) {
        // Compute the actual term index k, starting from 0
        int k = thread_id_global + idx * nthreads_total;
        if (k < N) {
            // Calculate (-1)^k
            double sign = (k % 2 == 0) ? 1.0 : -1.0;
            double denom = (2.0 * (double)k + 1.0);
            double val = sign / (denom * denom);
            sum[idx] = val;
        } else {
            sum[idx] = 0.0;
        }
    }
}

int main(int argc, char **argv) {
    int myid, nproc;
    double G_local = 0.0, G_global;
    double *sumHost, *sumDev;  // Pointers to host & device arrays

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // My MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); // Number of MPI processes

    omp_set_num_threads(NUM_DEVICE); // One OpenMP thread per GPU device

    int nthreads_total = nproc * NUM_DEVICE; // Total number of compute units

    #pragma omp parallel private(sumHost, sumDev) reduction(+:G_local)
    {
        int mpid = omp_get_thread_num(); // OpenMP thread ID (0-based)
        int thread_id_global = myid * NUM_DEVICE + mpid; // Global thread ID across MPI ranks

        // Set the CUDA device for this OpenMP thread
        cudaSetDevice(mpid % NUM_DEVICE);

        // Number of terms this particular thread will handle
        int nterms = (N - thread_id_global + nthreads_total - 1) / nthreads_total;
        if (nterms < 0) nterms = 0; // Safety check

        size_t size = nterms * sizeof(double);
        sumHost = (double *)malloc(size);
        cudaMalloc((void **)&sumDev, size);

        int threadsPerBlock = 256;
        int blocksPerGrid = (nterms + threadsPerBlock - 1) / threadsPerBlock;

        // Launch CUDA kernel to compute this thread's portion of Catalan's constant
        cal_G<<<blocksPerGrid, threadsPerBlock>>>(sumDev, thread_id_global, nthreads_total, nterms);

        // Copy results back to host
        cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

        // Partial sum on the CPU
        double partial_sum = 0.0;
        for (int i = 0; i < nterms; i++) {
            partial_sum += sumHost[i];
        }

        G_local += partial_sum;

        // Cleanup
        free(sumHost);
        cudaFree(sumDev);

        int dev_used;
        cudaGetDevice(&dev_used);

        // For debugging, print partial results
        printf("myid = %d; mpid = %d: device used = %d; partial G = %.15f\n", myid, mpid, dev_used, partial_sum);
    }

    // Reduce over all MPI processes
    MPI_Allreduce(&G_local, &G_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("Catalan's constant G ≈ %.15f\n", G_global);
    }

    MPI_Finalize();
    return 0;
}
