#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <cmath>
//#include <helper_cuda.h>


#define iteration(x) (j*x+i)    //iteration 1D array in 2 loops


__global__ void MatrixMulKernel(int* A, int* B, int* C, int Size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i < Size && j < Size)) {
		int Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < Size; ++k) {
			Pvalue += A[j * Size + k] * B[k * Size + i];
		}
		C[iteration(Size)] = Pvalue;
	}
}

__global__ void MatrixMulKernelShared(int* A, int* B, int* C, int blockSize, int Size) {
	extern __shared__ int sm[];

	int offset = blockSize * blockSize;
	int* a = sm;
	int* b = sm + offset;
	int* c = sm + 2 * offset;

	int I = blockIdx.x;
	int J = blockIdx.y;
	int K = blockIdx.z;

	int i = threadIdx.x;
	int j = threadIdx.y;

		

	//couting the position in matrixes
	
	int posA = (blockSize * J + j) * Size + K * blockSize + i;
	int posB = (blockSize * K + j) * Size + I * blockSize + i;
	int posC = (blockSize * J + j) * Size + I * blockSize + i;
	

	if (K * blockSize + i < Size && K * blockSize + j < Size) {
		//copy:
		a[iteration(blockSize)] = A[posA];
		b[iteration(blockSize)] = B[posB];
		c[iteration(blockSize)] = 0;
		__syncthreads();

		for (int k = 0; (k < blockSize && K * blockSize + k < Size ); ++k) {
			c[iteration(blockSize)] += a[j * blockSize + k] * b[k * blockSize + i];
		}
		//__syncthreads();
		//copy back:
		atomicAdd(C + posC, c[iteration(blockSize)]);

	}
}

int main()
{
	//Set up Matrixes
	const unsigned int Size = 31;


	unsigned int memSize = sizeof(int) * Size * Size;

	int* A, * B, * C;
	cudaMallocManaged(&A, memSize);
	cudaMallocManaged(&B, memSize);
	cudaMallocManaged(&C, memSize);


	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < Size; j++) {
			A[iteration(Size)] = 1; //for test I am using matrix of ones only because resulting matrix will have only Size numbers
			B[iteration(Size)] = 1;
		}
	}

	//matrix multiplication without using shared memory
	{
		unsigned int threadSize = Size, blockSize = 1;
		if (Size > 32) {
			threadSize = 32;
			blockSize = (unsigned int)ceil((float)Size / 32);
		}
		dim3 threadsPerBlock( threadSize, threadSize );
		dim3 blocksPerGrid( blockSize, blockSize );
		MatrixMulKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, Size);

		cudaDeviceSynchronize();

		bool Passed = true;
		for (int i = 0; i < Size; i++)
		{
			for (int j = 0; j < Size; j++)
			{
				if (C[iteration(Size)] != Size)
				{
					//std::cout << i << ", " << j<<": " << C[iteration(Size)] << std::endl;
					Passed = false;
					break;
				}
			}
			if (!Passed)	break;
		}
		if (Passed)
			std::cout << "Test passed" << std::endl;
		else
			std::cout << "Test failed" << std::endl;
	}

	//matrix multiplication using shared memory
	{
		std::cout << "========================\nNow using shared memory:" << std::endl;


		for (int i = 0; i < Size; i++) {
			for (int j = 0; j < Size; j++)
				C[iteration(Size)] = 0;
		}

		unsigned int threadSize;
		unsigned int blockSize;

		int dev = 0; // assume we have only 1 gpu
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		unsigned int SharedMem = deviceProp.sharedMemPerBlock / sizeof(int);
		threadSize = (unsigned int)sqrt(SharedMem / 3.f);	//we have to fit 3 matrixes
		threadSize = threadSize > 32 ? 32 : threadSize;
		threadSize = threadSize > Size ? Size : threadSize;

		//threadSize = 4;

		blockSize = ceil((float)Size / threadSize);

		std::cout << threadSize << "   " << blockSize << std::endl;

		dim3 threadsPerBlock( threadSize, threadSize );
		dim3 blocksPerGrid( blockSize,blockSize,blockSize );

		//std::cout << cudaGetErrorString(cudaGetLastError());
		MatrixMulKernelShared << < blocksPerGrid, threadsPerBlock, sizeof(int) * threadSize * threadSize * 3 >> > (A, B, C, threadSize, Size);

		

		cudaDeviceSynchronize();
		//std::cout << cudaGetErrorString(cudaGetLastError());

		bool Passed = true;
		for (int i = 0; i < Size; i++)
		{
			for (int j = 0; j < Size; j++)
			{
				//std::cout << C[iteration(Size)] << "  ";
				if (C[iteration(Size)] != Size)
				{
					Passed = false;
					break;
				}
			}
			//std::cout << std::endl;
			if (!Passed)	break;
		}
		if (Passed)
			std::cout << "Test passed" << std::endl;
		else
			std::cout << "Test failed" << std::endl;
	}


	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

	return 0;
}
