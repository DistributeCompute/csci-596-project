# Advanced Computational Approximations of Convergent Mathematical Series

## Introduction
This project harnesses the power of high-performance computing techniques, specifically parallel computation, to approximate both mathematical constants and the results of definite integrals using rapidly converging infinite series. Our main objective is to utilize the parallel computing capabilities of these technologies to achieve a high degree of precision within 1,000,000 terms of the series, with minimal computational overhead. Beyond this point, further terms only marginally increase precision, targeting extremely high decimal places. This makes our approach ideal for scenarios where high precision is critical but computational resources and time are at a premium.

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

- **Apéry's constant (ζ(3))**:

  <img width="239" alt="image" src="https://github.com/user-attachments/assets/9b611514-18ed-4e05-9ce0-66915b185cb2">



## Performance Comparison Table
Here is a table comparing the actual mathematical values of the constants with those obtained from our computational approximations:

| Constant                  | Actual Value            | Obtained Value         | Decimal Precision Achieved |
|---------------------------|-------------------------|------------------------|----------------------------|
| Euler-Mascheroni (γ)      | 0.57721566490153286061  | 0.57721566490153285378 | 16                         |
| Erdős–Borwein constant(E) | 1.60669515241529176378  | 1.60669515241529170524 | 16                         |
| Sierpiński's constant(K)  | 2.58498175957925321706  | 2.58493808352810106044 |  6                         |
| Landau-Ramanujan Constant | 0.76422365358922066299  | 0.76422365358922023743 | 15                         |
| Apéry's constant          | 1.20205690315959428539  | TBC                    | TBC                         |

  
## Installation
To use this project, follow these installation steps:

### Prerequisites
- An MPI implementation (e.g., MPICH, OpenMPI)
- CUDA Toolkit
- An OpenMP-compatible compiler


