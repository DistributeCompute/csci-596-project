// Hybrid MPI+CUDA computation of Khinchin’s constant K
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <math.h>

#define N 10000000    // Number of terms in the series (adjustable)
#define NUM_DEVICE 2  // Number of GPU devices = Number of OpenMP threads

// Probability function P(a=n) for partial quotients:
__device__ double p_of_n(int n) {
    // P(a=n) = log_2(1 + 1/(n(n+2)))
    //        = log(1 + 1/(n(n+2)))/log(2)
    double nn = (double)n;
    double val = 1.0 + 1.0/(nn*(nn+2.0));
    return log(val) / log(2.0);
}

__global__ void cal_logK(double *sum, int thread_id_global, int nthreads_total, int nterms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nterms) {
        int n = thread_id_global + 1 + idx * nthreads_total;
        if (n <= N) {
            double p = p_of_n(n);
            double ln_n = log((double)n);
            sum[idx] = p * ln_n;
        } else {
            sum[idx] = 0.0;
        }
    }
}


int main(int argc, char **argv) {
    int myid, nproc;
    double logK_local = 0.0, logK_global;
    double *sumHost, *sumDev;  // Pointers to host & device arrays

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // My MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); // Number of MPI processes

    omp_set_num_threads(NUM_DEVICE); // One OpenMP thread per GPU device

    int nthreads_total = nproc * NUM_DEVICE; // Total number of computing units

    #pragma omp parallel private(sumHost, sumDev) reduction(+:logK_local)
    {
        int mpid = omp_get_thread_num(); // OpenMP thread ID
        int thread_id_global = myid * NUM_DEVICE + mpid; // Global thread ID across all MPI ranks and OMP threads

        // Set the CUDA device for this OpenMP thread
        cudaSetDevice(mpid % NUM_DEVICE);

        // Calculate how many terms this particular thread will handle
        int nterms = (N - thread_id_global + nthreads_total - 1) / nthreads_total;
        if (nterms < 0) nterms = 0; // safety check

        size_t size = nterms * sizeof(double);
        sumHost = (double *)malloc(size);
        cudaMalloc((void **)&sumDev, size);

        int threadsPerBlock = 256;
        int blocksPerGrid = (nterms + threadsPerBlock - 1) / threadsPerBlock;

        // Launch CUDA kernel to compute partial sums for log(K)
        cal_logK<<<blocksPerGrid, threadsPerBlock>>>(sumDev, thread_id_global, nthreads_total, nterms);

        // Copy results back to host
        cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

        // Sum up the partial results
        double partial_sum = 0.0;
        for (int i = 0; i < nterms; i++)
            partial_sum += sumHost[i];

        // Accumulate into logK_local
        logK_local += partial_sum;

        // Cleanup
        free(sumHost);
        cudaFree(sumDev);

        int dev_used;
        cudaGetDevice(&dev_used);

        // For debugging, print partial results
        printf("myid = %d; mpid = %d: device used = %d; partial log(K) sum = %.15f\n", myid, mpid, dev_used, partial_sum);
    }

    // Reduce log(K) across all MPI processes
    MPI_Allreduce(&logK_local, &logK_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (myid == 0) {
        double K = exp(logK_global);
        printf("Khinchin’s constant K ≈ %.15f\n", K);
    }

    MPI_Finalize();
    return 0;
}
