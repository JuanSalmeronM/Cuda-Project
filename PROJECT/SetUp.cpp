#include"SetUp.h"

///we using the c++11 method to get random numbers becouse rand is bad
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(0.0, 1.0);
///

float RandomFloat(float a, float b) {
    float random = dist(mt);
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

void prepareData(float *data, int n, int *histogram, int hs, float dataBegin, float dataEnd)
{
	/*
	First we draw numbers to the histogram and then based on histogram we draw data
	*/
    float sum = 0;
    float *temp = new float [hs];
    for(int i=0;i<hs;i++)   //get the array with numbers between 0 and 1
    {
        temp[i] = dist(mt); 
        sum += temp[i];
    }

    float mult = n / sum;

    for(int i=0;i<hs;i++)   //now we transfer the temp to int value in sucha a way that:   sum of histogram = n
    {  
        histogram[i] = round(mult * temp[i]);  
    }

    delete temp;

    
    //testing   there is a problem with rounding so we do it to avoid errors
    int testSum=0;
    for(int i=0;i<hs;i++)
        testSum+=histogram[i];  
    if(testSum != n){   //checking if the sum of histogram is equal to n
        histogram[hs - 1] += n - testSum; //if test failed then we fix it by adding difference to the last column
    }



    float dataInterstice = dataEnd - dataBegin;
    float d_data = dataInterstice / hs;
    float dataB = dataBegin;

    //std::cout<<"dataInterstice: "<<dataInterstice<<"d_data: "<<d_data<<std::endl;

    int i=0;
    for(int j=0;j<hs;j++){
        for(int k=0;k<histogram[j];k++)
        {
            data[i] = RandomFloat(dataB, dataB + d_data);
            i++;
			//std::cout << i << std::endl;
        }
        dataB += d_data;
    }



}