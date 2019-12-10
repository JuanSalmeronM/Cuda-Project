#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <numeric>
#include <ctime>
using namespace std;

__global__ void SumReduction(int* input, int n)
{
	// Handle to thread block group
	extern __shared__ int sm[];

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sm[tid] = (i < n) ? input[i] : 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		if ((tid % (2 * s)) == 0)
		{
			sm[tid] += sm[tid + s];
		}

		__syncthreads();
	}

	// write result for this block to global mem
	//printf("%d: %d   , block ID: %d \n", threadIdx.x, sm[tid], blockIdx.x);
	if (tid == 0) input[blockIdx.x] = sm[0];
	
}

int main()
{
	unsigned int Size = 8887776;

	int* vec;
	cudaMallocManaged(&vec, Size * sizeof(int));			//allocating memory
	for (int i = 0; i < Size; i++) {			//writing only 1 to table becouse sum of it will be = Size easy to check
		vec[i] = 1;
	}
	//====================
	//STARTING CALCULATION

	
	//clock_t begin = clock();
	long long result = 0;		//we have to calculate result before kernel becouse kernel modify the vec
	for (int i = 0; i < Size; i++)
		result += vec[i];
	//clock_t end = clock();
	//double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	
	unsigned int maxThreads = 32;

	cudaEvent_t start, stop;		//time
	float time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	unsigned int threadsNum; 
	unsigned int gridSize = 2;
	int newSize = Size;

	while (gridSize > 1) {

		threadsNum = pow(2, ceil(log2(static_cast<float>(newSize))));		//calculatin threads per blocks - must be the power of two  and we want it to be as small as possible

		threadsNum = maxThreads < threadsNum ? maxThreads : threadsNum;		//chcecking if the threds amount are not too big

		gridSize = ceil(static_cast<float>(newSize) / threadsNum);			//calculating the blocks per grid;


		std::cout << "Block per grid: " << gridSize << "     Threads per block: " << threadsNum << std::endl;

		dim3 threadsPerBlock(threadsNum);
		dim3 blocksPerGrid(gridSize);

		size_t SharedSize = threadsNum * sizeof(int);		//calculating shared memory size

		
		cudaEventRecord(start);	//begin measure the time 
		SumReduction << <blocksPerGrid, threadsPerBlock, SharedSize >> > (vec, newSize);
		cudaEventRecord(stop);	//stop measure the time
		cudaEventSynchronize(stop);

		float milliseconds = 0;	
		cudaEventElapsedTime(&milliseconds, start, stop);	//the time output
		time += milliseconds;



		newSize = gridSize;				//now we have to sum firsts elements of every block sa newSize is now amount of blocks per grid

		cudaDeviceSynchronize();		//dont want to start new itteration befor GPU finish its job
		//std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;		//checking errors

	}

	

	if (vec[0] == result)	//checking if the results is proper (we add only 1 sa the results will be Size)
		std::cout << "Test passed! result = " << result <<"		Time GPU: "<<time <<" ms"<<std::endl;
	else
		std::cout << "Test failed! result =   ,   vec[0] = " << result<<vec[0]<< std::endl;
	//}


	cudaFree(vec);
	
	return 0;
}
