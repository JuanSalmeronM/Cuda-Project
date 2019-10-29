/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into B. The 2 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(int numElements, float *x, float *y)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		y[i] = x[i] + y[i];
	}
}



/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;


    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    // Print the vector length to be used, and compute its size
    unsigned long long numElements = 2<<29;
    size_t size = numElements * sizeof(float);

    std::cout<<deviceProp.name<<std::endl;

    std::cout<<"[Vector addition of "<<numElements<<" elements]\n";
    std::cout<<size<<std::endl;

    if(size > deviceProp.totalGlobalMem)
    {
    	std::cout<<"NOT ENOUGH MEMORY!\n Total memory: "<<deviceProp.totalGlobalMem<<std::endl;
    	return 0;
    }
    else
    {
    	std::cout<<"You got the memory :)\n";
    }


    ///
    float *x, *y;
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);
    ///

    // Initialize the host input vectors
    for (int i = 0; i < numElements; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
      }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(numElements, x, y);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();


    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < numElements; i++)
      maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    printf("Test PASSED\n");

    ///
    cudaFree(x);
    cudaFree(y);
    printf("Done\n");
    return 0;
}


