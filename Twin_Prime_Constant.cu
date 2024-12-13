// Hybrid MPI+CUDA computation of the Twin Prime Constant C2
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <math.h>
#include <string.h>

#define MAX_PRIME 20000000  // Upper bound for prime generation
#define NUM_DEVICE 2        // Number of GPU devices per MPI process

// CUDA kernel to compute partial sums of log(1 - 1/(p-1)^2)
__global__ void cal_logC2(double *dev_primes, int num_primes, double *dev_results, int thread_id_global, int nthreads_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int term_index = thread_id_global + idx * nthreads_total;
    if (term_index < num_primes) {
        double p = dev_primes[term_index];
        // For p >= 3:
        // term = (1 - 1/(p-1)^2)
        // log(term) = log(1 - 1/(p-1)^2)
        double denom = (p - 1.0);
        double factor = 1.0 - 1.0/(denom*denom);
        // Safeguard against potential floating errors:
        if (factor <= 0.0) {
            dev_results[idx] = 0.0; // Should not happen for large primes
        } else {
            dev_results[idx] = log(factor);
        }
    }
}

// Simple Sieve of Eratosthenes to generate primes up to MAX_PRIME
// Only run on rank 0, then broadcast.
int sieve_primes(int max_n, int **primes_out) {
    char *is_prime = (char*)malloc(max_n+1);
    memset(is_prime, 1, max_n+1);
    is_prime[0] = 0; is_prime[1] = 0;
    for (int i = 2; i*i <= max_n; i++) {
        if (is_prime[i]) {
            for (int j = i*i; j <= max_n; j+=i) {
                is_prime[j] = 0;
            }
        }
    }
    // Count primes
    int count = 0;
    for (int i=2; i<=max_n; i++) {
        if (is_prime[i]) count++;
    }
    *primes_out = (int*)malloc(count*sizeof(int));
    int idx = 0;
    for (int i=2; i<=max_n; i++) {
        if (is_prime[i]) (*primes_out)[idx++] = i;
    }
    free(is_prime);
    return count;
}

int main(int argc, char **argv) {
    int myid, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Generate primes on rank 0
    int *primes = NULL;
    int num_primes = 0;
    if (myid == 0) {
        printf("Generating primes up to %d...\n", MAX_PRIME);
        num_primes = sieve_primes(MAX_PRIME, &primes);
        printf("Number of primes found: %d\n", num_primes);
    }

    // Broadcast number of primes
    MPI_Bcast(&num_primes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_primes == 0) {
        // No primes? Something went wrong or MAX_PRIME too small.
        MPI_Finalize();
        return 0;
    }

    // Broadcast prime list
    if (myid != 0) {
        primes = (int*)malloc(num_primes*sizeof(int));
    }
    MPI_Bcast(primes, num_primes, MPI_INT, 0, MPI_COMM_WORLD);

    // We only consider primes >= 3 for twin prime constant
    // Find the starting index (first prime >=3 is at primes[1] if first prime is 2)
    int start_index = 0;
    while (start_index < num_primes && primes[start_index] < 3) start_index++;
    int effective_num_primes = num_primes - start_index;
    if (effective_num_primes < 1) {
        if (myid == 0) printf("No primes >=3 found!\n");
        free(primes);
        MPI_Finalize();
        return 0;
    }

    // Convert to double for GPU
    double *primes_double = (double*)malloc(effective_num_primes*sizeof(double));
    for (int i = 0; i < effective_num_primes; i++)
        primes_double[i] = (double)primes[start_index + i];

    // Free the int primes array if desired
    free(primes);

    int nthreads_total = nproc * NUM_DEVICE;
    double logC2_local = 0.0, logC2_global;

    omp_set_num_threads(NUM_DEVICE);

    #pragma omp parallel reduction(+:logC2_local)
    {
        int mpid = omp_get_thread_num();
        int thread_id_global = myid * NUM_DEVICE + mpid;

        cudaSetDevice(mpid % NUM_DEVICE);

        // Determine how many terms this thread will handle
        int nterms = (effective_num_primes - thread_id_global + nthreads_total - 1) / nthreads_total;
        if (nterms < 0) nterms = 0;

        size_t size = nterms * sizeof(double);
        double *sumHost = (double*)malloc(size);
        double *dev_primes, *dev_results;

        // Allocate on device
        cudaMalloc((void**)&dev_primes, nterms*sizeof(double));
        cudaMalloc((void**)&dev_results, size);

        // Extract the subset of primes this thread handles
        // Interleaving: term_index = thread_id_global + idx*nthreads_total
        // idx runs from 0 to nterms-1
        double *primes_subset = (double*)malloc(size);
        for (int i = 0; i < nterms; i++) {
            int global_idx = thread_id_global + i*nthreads_total;
            primes_subset[i] = primes_double[global_idx];
        }

        // Copy subset to device
        cudaMemcpy(dev_primes, primes_subset, size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (nterms + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel to compute partial sums of logs
        cal_logC2<<<blocksPerGrid, threadsPerBlock>>>(dev_primes, nterms, dev_results, 0, 1);
        // Note: We passed thread_id_global and nthreads_total logic into how we formed subset.
        // Here it's simplified: we directly formed the subset for each thread.

        cudaMemcpy(sumHost, dev_results, size, cudaMemcpyDeviceToHost);

        double partial_sum = 0.0;
        for (int i=0; i<nterms; i++)
            partial_sum += sumHost[i];

        logC2_local += partial_sum;

        int dev_used;
        cudaGetDevice(&dev_used);

        printf("myid = %d; mpid = %d: device used = %d; partial log(C2) sum = %.15f\n", myid, mpid, dev_used, partial_sum);

        free(sumHost);
        free(primes_subset);
        cudaFree(dev_primes);
        cudaFree(dev_results);
    }

    MPI_Allreduce(&logC2_local, &logC2_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (myid == 0) {
        double C2 = exp(logC2_global);
        printf("Twin Prime Constant C2 ≈ %.15f\n", C2);
    }

    free(primes_double);
    MPI_Finalize();
    return 0;
}
