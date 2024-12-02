# Advanced Computational Approximations of Convergent Mathematical Series

## Introduction
This project harnesses the power of high-performance computing techniques, specifically MPI, CUDA, and OpenMP, to approximate both mathematical constants and the results of definite integrals using rapidly converging infinite series. Our main objective is to utilize the parallel computing capabilities of these technologies to achieve a high degree of precision within 10,000 terms of the series, with minimal computational overhead. Beyond this point, further terms only marginally increase precision, targeting extremely high decimal places. This makes our approach ideal for scenarios where high precision is critical but computational resources and time are at a premium.

The selection of series and integrals for this project is carefully curated to ensure that each can be broken down effectively into a component series that converges swiftly. This approach not only facilitates efficient computation but also allows for the exploration of intricate mathematical relationships and properties in a computationally feasible manner. By focusing on both constants known for their fast convergence and integrals that can similarly be approximated by series, this project caters to a wide range of scientific and engineering applications, offering tools to solve complex problems where precision is paramount.

## Supported Constants and Integrals
Here are some examples of constants and definite integrals, along with their formulas and series approximations, that our project can efficiently approximate:

- **Euler's Constant (e)**:

  <img width="255" alt="image" src="https://github.com/user-attachments/assets/6194d03a-cb59-4961-ab47-1e75a915dc7b">

- **The Euler-Mascheroni Constant (γ)**:

  <img width="255" alt="image" src="https://github.com/user-attachments/assets/b701f436-11e3-4cb8-a312-26499ce1e50c">
- **Landau-Ramanujan Constant (K)**:

  <img width="255" alt="image" src="https://github.com/user-attachments/assets/7cf668cd-d3c8-4e5b-9c55-04adfe7e1bad">
  
## Installation
To use this project, follow these installation steps:

### Prerequisites
- An MPI implementation (e.g., MPICH, OpenMPI)
- CUDA Toolkit
- An OpenMP-compatible compiler

