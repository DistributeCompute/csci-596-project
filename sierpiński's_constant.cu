// Hybrid MPI+CUDA computation of Sierpiński's constant K
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <math.h>

#define N 1000000  // Number of terms in the series (adjustable)

#define NUM_DEVICE 2 // Number of GPU devices = Number of OpenMP threads

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA Kernel to compute partial sums for Sierpiński's constant K
__global__ void cal_K(double *sum, int thread_id_global, int nthreads_total, int nterms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
    if (idx < nterms) {
        int k = thread_id_global + 1 + idx * nthreads_total;  // Interleaved term index
        if (k <= N) {
            int r2 = 0;
            int sqrt_k = (int)sqrt((double)k);
            for (int a = 0; a <= sqrt_k; a++) {
                int a2 = a * a;
                int b2 = k - a2;
                if (b2 >= 0) {
                    int b = (int)sqrt((double)b2);
                    if (b >= a && b * b == b2) {
                        if (a == 0 && b == 0)
                            r2 += 1;  // Only (0,0)
                        else if (a == 0 || b == 0)
                            r2 += 4;
                        else if (a == b)
                            r2 += 4;
                        else
                            r2 += 8;
                    }
                }
            }
            sum[idx] = (double)r2 / (double)k;
        } else {
            sum[idx] = 0.0;
        }
    }
}

int main(int argc, char **argv) {
    int myid, nproc;
    double K = 0.0, Kg;
    double *sumHost, *sumDev;  // Pointers to host & device arrays
    int dev_used;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // My MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);  // Number of MPI processes

    omp_set_num_threads(NUM_DEVICE); // One OpenMP thread per GPU device

    int nthreads_total = nproc * NUM_DEVICE; // Total number of compute units

    #pragma omp parallel private(sumHost, sumDev, dev_used) reduction(+:K)
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
        cal_K <<<blocksPerGrid, threadsPerBlock>>> (sumDev, thread_id_global, nthreads_total, nterms);

        // Copy the results back to host
        cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

        // Sum up the partial results
        double partial_K = 0.0;
        for (int i = 0; i < nterms; i++)
            partial_K += sumHost[i];

        // Accumulate the partial sums into the total K
        K += partial_K;

        // CUDA cleanup
        free(sumHost);
        cudaFree(sumDev);
        cudaGetDevice(&dev_used);

        // Output the partial sum for this thread
        printf("myid = %d; mpid = %d: device used = %d; partial K = %.20f\n", myid, mpid, dev_used, partial_K);
    } // End of OpenMP parallel region

    // Reduction over MPI processes
    MPI_Allreduce(&K, &Kg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Compute π ln N
    double pi_ln_N = M_PI * log((double)N);

    // Compute Sierpiński's constant K
    double K_value = Kg - pi_ln_N;

    if (myid == 0) printf("Sierpiński's constant K ≈ %.20f\n", K_value);

    MPI_Finalize();
    return 0;
}
