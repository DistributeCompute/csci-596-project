// Hybrid MPI+CUDA computation of the Landau-Ramanujan constant b
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define N 1000000  // Upper limit for prime numbers (adjustable)

#define NUM_DEVICE 2 // Number of GPU devices = Number of OpenMP threads

// Function to generate primes p ≡ 3 mod 4 up to N
void generate_primes(int **primes, int *num_primes) {
    bool *is_prime = (bool *)malloc((N + 1) * sizeof(bool));
    for (int i = 0; i <= N; i++)
        is_prime[i] = true;

    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i * i <= N; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= N; j += i)
                is_prime[j] = false;
        }
    }

    // Count the number of primes p ≡ 3 mod 4
    int count = 0;
    for (int i = 2; i <= N; i++) {
        if (is_prime[i] && i % 4 == 3)
            count++;
    }

    *primes = (int *)malloc(count * sizeof(int));
    *num_primes = count;

    int idx = 0;
    for (int i = 2; i <= N; i++) {
        if (is_prime[i] && i % 4 == 3)
            (*primes)[idx++] = i;
    }

    free(is_prime);
}

// CUDA Kernel to compute partial logarithms for the Landau-Ramanujan constant b
__global__ void compute_partial_logs(double *partial_logs, int *primes_dev, int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        int p = primes_dev[idx];
        double log_term = -0.5 * log(1.0 - 1.0 / (p * p));
        partial_logs[idx] = log_term;
    }
}

int main(int argc, char **argv) {
    int myid, nproc;
    double log_b_local = 0.0, log_b_global;
    int *primes = NULL;
    int num_primes;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); // Number of MPI processes

    if (myid == 0) {
        // Generate primes on rank 0
        generate_primes(&primes, &num_primes);
    }

    // Broadcast the number of primes to all processes
    MPI_Bcast(&num_primes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute the primes array to all processes
    if (myid != 0) {
        primes = (int *)malloc(num_primes * sizeof(int));
    }
    MPI_Bcast(primes, num_primes, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine the range of primes for this process
    int primes_per_proc = (num_primes + nproc - 1) / nproc;
    int start_idx = myid * primes_per_proc;
    int end_idx = start_idx + primes_per_proc;
    if (end_idx > num_primes)
        end_idx = num_primes;

    int local_num_primes = end_idx - start_idx;

    // Allocate memory for partial logs
    double *partial_logs = (double *)malloc(local_num_primes * sizeof(double));

    omp_set_num_threads(NUM_DEVICE); // Set number of OpenMP threads

    #pragma omp parallel reduction(+:log_b_local)
    {
        int omp_id = omp_get_thread_num(); // OpenMP thread ID
        int dev_id = omp_id % NUM_DEVICE;

        // Set the CUDA device for this thread
        cudaSetDevice(dev_id);

        int threadsPerBlock = 256;

        // Divide the primes among OpenMP threads
        int primes_per_thread = (local_num_primes + NUM_DEVICE - 1) / NUM_DEVICE;
        int thread_start = start_idx + omp_id * primes_per_thread;
        int thread_end = thread_start + primes_per_thread;
        if (thread_end > end_idx)
            thread_end = end_idx;

        int thread_num_primes = thread_end - thread_start;

        if (thread_num_primes > 0) {
            // Allocate device memory
            int *primes_dev;
            double *partial_logs_dev;
            size_t size_int = thread_num_primes * sizeof(int);
            size_t size_double = thread_num_primes * sizeof(double);

            cudaMalloc((void **)&primes_dev, size_int);
            cudaMalloc((void **)&partial_logs_dev, size_double);

            // Copy primes to device
            cudaMemcpy(primes_dev, &primes[thread_start], size_int, cudaMemcpyHostToDevice);

            // Calculate grid and block dimensions
            int blocksPerGrid = (thread_num_primes + threadsPerBlock - 1) / threadsPerBlock;

            // Launch the CUDA kernel
            compute_partial_logs<<<blocksPerGrid, threadsPerBlock>>>(partial_logs_dev, primes_dev, thread_num_primes);

            // Copy results back to host
            cudaMemcpy(&partial_logs[thread_start - start_idx], partial_logs_dev, size_double, cudaMemcpyDeviceToHost);

            // Sum the logarithms of the terms
            double thread_log_sum = 0.0;
            for (int i = thread_start - start_idx; i < thread_end - start_idx; i++) {
                thread_log_sum += partial_logs[i];
            }

            // Accumulate the sum into the reduction variable
            log_b_local += thread_log_sum;

            // Free device memory
            cudaFree(primes_dev);
            cudaFree(partial_logs_dev);
        }
    } // End of OpenMP parallel region

    // Reduce the partial log sums from all processes
    MPI_Allreduce(&log_b_local, &log_b_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (myid == 0) {
        // Compute the final value of b
        double b_global = exp(log_b_global) * (1.0 / sqrt(2.0));
        printf("Landau-Ramanujan constant b ≈ %.20f\n", b_global);
    }

    free(primes);
    free(partial_logs);

    MPI_Finalize();
    return 0;
}
