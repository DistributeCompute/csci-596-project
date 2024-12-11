include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <math.h>

#define N 100000000  // Number of terms in the harmonic series
#define NUM_DEVICE 2 // Number of GPU devices = Number of OpenMP threads

// CUDA kernel to compute partial harmonic series
__global__ void harmonic_sum(double *sum, int thread_id_global, int nthreads_total, int nterms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
    if (idx < nterms) {
        int k = thread_id_global + 1 + idx * nthreads_total;  // Interleaved term index
        if (k <= N) {
            sum[idx] = 1.0 / k;  // Compute 1/k
        } else {
            sum[idx] = 0.0;  // Handle out-of-bound terms
        }
    }
}

int main(int argc, char **argv) {
    int myid, nproc;
    double harmonic_sum_local = 0.0, harmonic_sum_global, euler_mascheroni_constant;
    double *sumHost, *sumDev;  // Pointers to host & device arrays
    int dev_used;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // My MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);  // Number of MPI processes

    omp_set_num_threads(NUM_DEVICE); // One OpenMP thread per GPU device

    int nthreads_total = nproc * NUM_DEVICE; // Total number of compute units

    #pragma omp parallel private(sumHost, sumDev, dev_used) reduction(+:harmonic_sum_local)
    {
        int mpid = omp_get_thread_num(); // OpenMP thread ID
        int thread_id_global = myid * NUM_DEVICE + mpid; // Global thread ID

        // Set the CUDA device for the thread
        cudaSetDevice(mpid % NUM_DEVICE);

        // Calculate the number of terms this thread will process
        int nterms = (N - thread_id_global + nthreads_total - 1) / nthreads_total;

        size_t size = nterms * sizeof(double);  // Memory size for this thread
        sumHost = (double *)malloc(size);       // Allocate host memory
        cudaMalloc((void **)&sumDev, size);     // Allocate device memory

        // Adjust the number of threads and blocks based on nterms
        int threadsPerBlock = 256;
        int blocksPerGrid = (nterms + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the CUDA kernel
        harmonic_sum <<<blocksPerGrid, threadsPerBlock>>> (sumDev, thread_id_global, nthreads_total, nterms);

        // Copy the results back to host
        cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

        // Sum up the partial results
        double partial_sum = 0.0;
        for (int i = 0; i < nterms; i++)
            partial_sum += sumHost[i];
	        // Accumulate the partial sums into the local total
        harmonic_sum_local += partial_sum;

        // CUDA cleanup
        free(sumHost);
        cudaFree(sumDev);
        cudaGetDevice(&dev_used);

        // Output the partial sum for this thread
        printf("myid = %d; mpid = %d: device used = %d; partial sum = %.20f\n", myid, mpid, dev_used, partial_sum);
    } // End of OpenMP parallel region

    // Reduction over MPI processes
    MPI_Allreduce(&harmonic_sum_local, &harmonic_sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Compute the Euler-Mascheroni constant
    if (myid == 0) {
        double lnN = log((double)N);
        euler_mascheroni_constant = harmonic_sum_global - lnN;
        printf("Euler-Mascheroni Constant (γ) ≈ %.20f\n", euler_mascheroni_constant);
    }

    MPI_Finalize();
    return 0;
}

