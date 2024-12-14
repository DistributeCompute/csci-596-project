# Computational Approximations of Monotonic Convergent Series

## Introduction
This project harnesses the power of parallel computing techniques, to approximate both mathematical constants and the results of definite integrals using rapidly converging infinite series. Our main objective is to utilize the parallel computing capabilities to achieve a high degree of precision within say 1,000,000 terms (For now. We plan to increase the number of terms to a much higher value to increase precision) of the series, with minimal computational overhead. Beyond this point, further terms only marginally increase precision, targeting extremely high decimal places. This makes our approach ideal for scenarios where high precision is critical but computational resources and time are at a premium.

The selection of series and integrals for this project is carefully curated to ensure that each can be broken down effectively into a component series that converges relatively quickly. This approach not only facilitates efficient computation but also allows for the exploration of  mathematical relationships and properties in a computationally feasible manner. We focus on both constants known for their fast convergence and integrals that can similarly be approximated to a certain precision by series.

A key aspect of this project is the identification and exploitation of iterative patterns within the chosen series. Many mathematical constants and integrals can be expressed as series with recurring structures or predictable term-to-term relationships. By recognizing these patterns, we can:
* Develop efficient algorithms that capitalize on the series' inherent structure.
* Distribute computations across multiple processors, significantly accelerating the calculation process.
* Achieve high precision results in a fraction of the time required by traditional sequential methods.

## Supported Constants and Integrals
Here are some examples of constants and definite integrals, along with their formulas and series approximations, that our project can efficiently approximate:

- **Erdős–Borwein constant (E)**:

  <img width="255" alt="image" src="https://github.com/user-attachments/assets/44f84038-d244-4c49-9f0e-7bfe95f87cb9">

- **Sierpiński's constant (K)**:

  <img width="255" alt="image" src="https://github.com/user-attachments/assets/eee0ac9d-906c-4c6c-9802-0d0c7cb52fb8">

  Here, `r_2(k)` represents the number of ways the integer `k` can be expressed as the sum of two squares, that is, `k = a^2 + b^2`, where `a` and `b` are integers. This definition leverages the analytical properties of number theory, particularly the distribution of numbers expressible in certain quadratic forms, to define the constant.

  
- **Landau-Ramanujan Constant (b)**:

  <img width="326" alt="image" src="https://github.com/user-attachments/assets/c39533af-6f57-414e-a24c-7b8c78d76e1f">
  
  This product runs over all prime numbers p that are congruent to 3 modulo 4.

- **Catalan's Constant (G)**:

  <img width="326" alt="image" src="https://github.com/user-attachments/assets/fd38cada-30d7-4645-9d1a-15ccc93b98a3">

- **Khinchin's Constant**:

  <img width="239" alt="image" src="https://github.com/user-attachments/assets/56b5206d-28f1-4648-9cd1-37a48ac61a40">

- **Twin Prime Constant**:

  <img width="239" alt="image" src="https://github.com/user-attachments/assets/10e6a8cc-94cc-47a7-8c4b-34acd4c5bd48">

- **Apéry's constant (ζ(3))**:

  <img width="239" alt="image" src="https://github.com/user-attachments/assets/9b611514-18ed-4e05-9ce0-66915b185cb2">

- **Euler-Mascheroni constant (γ)**:
  
  <img width="239" alt="image" src="https://github.com/user-attachments/assets/fe52444e-6fda-4c10-abd5-68358cdb5406">

  where ζ(k) is the Riemann zeta function. But this can be alternatively expressed as below. 

  <img width="256" alt="image" src="https://github.com/user-attachments/assets/d790a956-2e57-4596-a3d5-fe50069e52a5">


  This form is more easier to "Divide and Conquer" compared to the first form.

## Example Kernel (Sierpiński's constant)

```cpp
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
                            r2 += 1;
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
```


## Current Precision Comparison Table
Here is a table comparing the actual mathematical values of the constants with those obtained from our computational approximations:

| Constant                  | Actual Value            | Obtained Value         | Decimal Precision Achieved |
|---------------------------|-------------------------|------------------------|----------------------------|
| Euler-Mascheroni (γ)      | 0.57721566490153286061  | 0.57721566990024442134 |  8                         |
| Erdős–Borwein constant(E) | 1.60669515241529176378  | 1.60669515241529170524 | 16                         |
| Sierpiński's constant(K)  | 2.58498175957925321706  | 2.58493808352810106044 |  6                         |
| Landau-Ramanujan Constant | 0.76422365358922066299  | 0.76422364064766579173 |  7                         |
| Khinchin’s constant K     | 2.685452001065306       | 2.685445368968978      |  4                         |
| Catalan's constant G      | 0.915965594177219       | 0.915965594177080      |  12                        |
| Twin Prime Constant C2    |0.660161815846869        | 0.660161817705094      |  8                         |
| Apéry's constant          | 1.20205690315959428539  | TBC                    | TBC                        |
| Arctan(1)                 | 0.7853981633974483      | 0.785397               | 6                          |


## Steps to Compile and Run the Code

1. **Clear any previously loaded modules:**
   module purge
2. **Load the required modules:**
   module load usc/8.3.0
   module load cuda
3. **Compile the program using NVCC:**
  nvcc -Xcompiler -fopenmp ${CONSTANT_NAME}.cu -o ${CONSTANT_NAME} \
  -I${OPENMPI_ROOT}/include -L${OPENMPI_ROOT}/lib -lmpi -lgomp
4. **Submit the job with sbatch:**
   sbatch constant.sl
5. **View the Output:**
   more ${CONSTANT_NAME}.out

   






