/*
There can be problem with crashing app
It is caused by WDDM TDR delay
this delay works in such a way that kill the kernel if it doesnt finish in specific time
so for big numbers it can be a problem 
but you can change time or even turn it off in Nsight monitor : option->general->microsoft display driver
*/
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SetUp.h"	//preparing data


#define PI 3.14159265358979323846


#define N	10000	//data size
#define ES	10000	//estimation size
#define HS	20		//histogram size	the lower hs is the better results will appear
//do not spoil and dont set data size greater than histogram size

__global__ void estimationKernel(float* data, size_t n, float* kernelEstimation, size_t es, float dx, float h)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < es; i += stride)
	{
		float di;
		di = dx * i;	//on which position on OX axis we calculate the estimation

		kernelEstimation[i] = 0;
		for (int j = 0; j < n; j++)
		{
			//formula:
			float power = -0.5f * (di - data[j]) * (di - data[j]) / h / h;
			kernelEstimation[i] += expf(power);
			
		}
		kernelEstimation[i] /= (n * h) * sqrt(2 * PI);	//also formula

	}
}

void ErrorCheck() {		//cudas error checker
	cudaError_t Err = cudaGetLastError();
	if (Err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(Err));
}

int main()
{
	float dataBegin = 0.0f;	//data range
	float dataEnd = 100.0f;	//^
	
	int hist[HS];	//histogram, it will be the nice way to show results

	float *data;	//input data
	float *kernelEstimation;	//output data

	cudaMallocManaged(&data, N * sizeof(float));
	cudaMallocManaged(&kernelEstimation, ES * sizeof(float));

	cudaDeviceSynchronize();
	ErrorCheck();

	//cudaMemPrefetchAsync(data, N, cudaCpuDeviceId);	//causing stange error
	ErrorCheck();

	std::ofstream kerEstFile;	//files we write output and histogram
	std::ofstream histFile;
	kerEstFile.open("estimation.txt");
	histFile.open("histogram.txt");

	
	//Preparing data
	prepareData(data, N, hist, HS, dataBegin, dataEnd);

	

	//Expected value:
	float ev = 0;
	for (int i = 0; i < N; i++)
		ev += data[i];
	ev /= N;

	//Standard deviation:
	float std = 0;
	for (int i = 0; i < N; i++)
		std += (pow(data[i] - ev, 2)) / N;
	std = sqrt(std);

	std::cout << "Expected value: " << ev << "\nStandard deviation: " << std << std::endl;



	float h = 1.06 * std * pow(N, -0.2);  //this formula for h is best for Gaussian basis functions but you can also set it to ny valu for exapmle 1.5f
	std::cout << "h = " << h << std::endl;
	

	float dx = (dataEnd - dataBegin) / ES; //distance between 2 points, we calculate valu to them
	//std::cout << "Dx: " << dx << std::endl;
					


	//kernel estimation GPU

	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	ErrorCheck();

	//cudaMemPrefetchAsync(data, N, deviceId);	//causing stange error
	//cudaMemPrefetchAsync(kernelEstimation, ES, deviceId);
	ErrorCheck();

	size_t threadsPerBlock;
	size_t numberOfBlocks;

	threadsPerBlock = 256;
	numberOfBlocks = 32 * numberOfSMs;

	std::cout << "threadsPerBlock: " << threadsPerBlock << ", numberOfBlocks: " << numberOfBlocks << std::endl;

	//TIME:
	//we use cuda events to calculate time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start);

	estimationKernel << <numberOfBlocks, threadsPerBlock >> > (data, N, kernelEstimation, ES, dx, h);

	cudaEventRecord(stop);

	cudaDeviceSynchronize();

	ErrorCheck();


	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "TIME: " << milliseconds << " ms" << std::endl;

	/*/////
	double calk = 0.0f;
	for(int i=0;i<ES;i++)
	{
		calk += kernelEstimation[i];
	}
	std::cout << "calka = " << calk * dx << std::endl;
	////*/
	//calculating Mean squared error	it shows how far are we from the histogram
	float err = 0;
	if (ES >= HS) {
		int di = round(ES / HS);
		int j = 0;
		for (int i = di/2.0f; i < ES; i += di) {
			err += pow(hist[j] / N - kernelEstimation[i],2);
		}
		std::cout << "Mean squared error: " << err / HS << std::endl;
	}
	else {
		int di = round(HS / ES);
		int j = 0;
		for (int i = 0; i < HS; i += di) {
			err += pow(hist[i] / N - kernelEstimation[j], 2);
		}
		std::cout << "Mean squared error1111: " << err / ES << std::endl;
	}
	

	//Saving output in file

	std::cout<<"Saving output"<<std::endl;

	for (int i = 0; i < HS; i++)
		histFile << hist[i] << std::endl;
	
	for (int i = 0; i < ES; i++)
		kerEstFile << kernelEstimation[i] << std::endl;
	

	cudaFree(data);
	cudaFree(kernelEstimation);

	//system("pause");
	return 0;
}
