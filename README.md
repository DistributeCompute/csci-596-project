# High Precision Constant and Integral Approximation

## Introduction
This project harnesses the power of high-performance computing techniques, specifically MPI, CUDA, and OpenMP, to approximate both mathematical constants and the results of definite integrals using rapidly converging infinite series. Our main objective is to utilize the parallel computing capabilities of these technologies to achieve a high degree of precision within 10,000 terms of the series, with minimal computational overhead. Beyond this point, further terms only marginally increase precision, targeting extremely high decimal places. This makes our approach ideal for scenarios where high precision is critical but computational resources and time are at a premium.

The selection of series and integrals for this project is carefully curated to ensure that each can be broken down effectively into a component series that converges swiftly. This approach not only facilitates efficient computation but also allows for the exploration of intricate mathematical relationships and properties in a computationally feasible manner. By focusing on both constants known for their fast convergence and integrals that can similarly be approximated by series, this project caters to a wide range of scientific and engineering applications, offering tools to solve complex problems where precision is paramount.

## Supported Constants and Integrals
Here are some examples of constants and definite integrals, along with their formulas and series approximations, that our project can efficiently approximate:

- **Euler's Constant (e)**:
  - **Formula**: γ = lim (n → ∞) ( ∑(k=1 to n) 1/k - log(n) )

- **The Euler-Mascheroni Constant (γ)**:
  - **Formula**: \( \gamma = \lim_{n \to \infty} \left( \sum_{k=1}^{n} \frac{1}{k} - \log(n) \right) \)

- **Apéry's Constant (ζ(3))**:
  - **Formula**: \( \zeta(3) = \sum_{n=1}^{\infty} \frac{1}{n^3} \)

- **Catalan's Constant (G)**:
  - **Formula**: \( G = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)^2} \)

- **Definite Integral Example**: Approximation of \( \int_0^1 \frac{\sin(x)}{x} \, dx \)
  - **Series Approximation**: Using a series derived from the Taylor expansion of \( \sin(x) \), we can compute the integral by approximating it as \( \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} \int_0^1 x^{2n} \, dx \).

## Installation
To use this project, follow these installation steps:

### Prerequisites
- An MPI implementation (e.g., MPICH, OpenMPI)
- CUDA Toolkit
- An OpenMP-compatible compiler

