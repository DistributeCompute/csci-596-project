// DistributeCompute: Parallel Computation using MPI and CUDA

// Hybrid MPI+CUDA computation of Pi
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <math.h>

#define NUM_DEVICE 2 // # of GPU devices = # of OpenMP threads
#define NBIN  100000000  // Number of bins, increasing for accuracy

__global__ void cal_apery(double *sum, int thread_id_global, int nthreads_total, int nterms)) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global index 
    if (idx < nterms) {
        // Compute the actual term index n for this thread
        int n = thread_id_global + 1 + idx * nthreads_total;
        if (n <= N) {
            double dn = (double)n;
            sum[idx] = 1.0/(dn*dn*dn);
        }
    }
}

// Kernel that executes on the CUDA device
__global__ void cal_apery(float *sum, int nbin, int offset, int nthreads, int nblocks) {
    int i;
    float n;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
    for (i=idx; i<nbin; i+=nthreads*nblocks) {  // Interleaved bin assignment to threads
        n = offset + i + 1.0;  // Start from 1 to avoid division by zero
        sum[idx] += 1.0/(n*n*n);  // Computing 1/n^3
    }
}

// int main(int argc,char **argv) {
// 	int myid,nproc,nbin,tid;
// 	float step,offset,pi=0.0,pig;
	
// 	float apery=0.0, apery_final;

// 	dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions (only use 1D)
// 	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions (only use 1D)
// 	float *sumHost,*sumDev;  // Pointers to host & device arrays
// 	int dev_used;

// 	MPI_Init(&argc,&argv);
// 	MPI_Comm_rank(MPI_COMM_WORLD,&myid);  // My MPI rank
// 	MPI_Comm_size(MPI_COMM_WORLD,&nproc);  // Number of MPI processes
// 	// nbin = NBIN/nproc;  // Number of bins per MPI process
// 	// step = 1.0/(float)(nbin*nproc);  // Step size with redefined number of bins
// 	// offset = myid*step*nbin;  // Quadrature-point offset

// 	omp_set_num_threads(NUM_DEVICE); // One OpenMP thread per GPU device
// 	nbin = NBIN/(nproc*NUM_DEVICE); // # of bins per OpenMP thread
// 	step = 1.0/(float)(nbin*nproc*NUM_DEVICE);

// 	#pragma omp parallel private(offset, sumHost, sumDev, tid, dev_used) reduction(+:pi)
// 	{	
// 		int mpid = omp_get_thread_num();
// 		offset = (NUM_DEVICE*myid+mpid)*step*nbin; // Quadrature-point offset
// 		cudaSetDevice(mpid%2);

// 		// cudaSetDevice(myid%2);
// 		size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size
// 		sumHost = (float *)malloc(size);  //  Allocate array on host
// 		cudaMalloc((void **) &sumDev,size);  // Allocate array on device
// 		cudaMemset(sumDev,0,size);  // Reset array in device to 0
// 		// Calculate on device (call CUDA kernel)
// 		cal_pi <<<dimGrid,dimBlock>>> (sumDev,nbin,step,offset,NUM_THREAD,NUM_BLOCK);
// 		// Retrieve result from device and store it in host array
// 		cudaMemcpy(sumHost,sumDev,size,cudaMemcpyDeviceToHost);
// 		// Reduction over CUDA threads
// 		for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
// 			pi += sumHost[tid];
// 		pi *= step;
// 		// CUDA cleanup
// 		free(sumHost);
// 		cudaFree(sumDev);
// 		cudaGetDevice(&dev_used);
// 		printf("myid = %d; mpid = %d: device used = %d; partial pi = %f\n", myid, mpid, dev_used, pi);
// 	}
	
// 	// Reduction over MPI processes
// 	MPI_Allreduce(&pi,&pig,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
// 	if (myid==0) printf("Apery's con = %f\n",pig);

// 	MPI_Finalize();
// 	return 0;
// }

int main(int argc, char **argv) {
    int myid, nproc, nbin, tid;
    int offset;
    float apery=0.0, apery_final;
    dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions (only use 1D)
    dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions (only use 1D)
    float *sumHost,*sumDev;  // Pointers to host & device arrays
    int dev_used;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);  // My MPI rank
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);  // Number of MPI processes

    omp_set_num_threads(NUM_DEVICE); // One OpenMP thread per GPU device
    nbin = NBIN/(nproc*NUM_DEVICE); // # of terms per OpenMP thread
	
    #pragma omp parallel private(offset, sumHost, sumDev, tid, dev_used) reduction(+:apery)
    {   
        int mpid = omp_get_thread_num();
        offset = (NUM_DEVICE*myid+mpid)*nbin; // Term offset for this thread

        cudaSetDevice(mpid%2);

        size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size
        sumHost = (float *)malloc(size);  //  Allocate array on host
        cudaMalloc((void **) &sumDev,size);  // Allocate array on device
        cudaMemset(sumDev,0,size);  // Reset array in device to 0

        // Calculate on device (call CUDA kernel)
        cal_apery <<<dimGrid,dimBlock>>> (sumDev,nbin,offset,NUM_THREAD,NUM_BLOCK);

        // Retrieve result from device and store it in host array
        cudaMemcpy(sumHost,sumDev,size,cudaMemcpyDeviceToHost);

        // Reduction over CUDA threads
        for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
            apery += sumHost[tid];

        // CUDA cleanup
        free(sumHost);
        cudaFree(sumDev);
        cudaGetDevice(&dev_used);
        printf("myid = %d; mpid = %d: device used = %d; partial sum = %f\n", 
               myid, mpid, dev_used, apery);
    }
    
    // Reduction over MPI processes
    MPI_Allreduce(&apery,&apery_final,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    if (myid==0) printf("Apery's constant = %f\n",apery_final);

    MPI_Finalize();
    return 0;
}