#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

constexpr int MAX_LEN = 30;

void radix_sort_cuda(char strArr[][MAX_LEN], int N) {

    // First we have to copy these data to device.
    size_t data_size = N * MAX_LEN;
    char* device_input;
    char* device_output;
    size_t pitch;

    if(pitch == MAX_LEN) cout << "HELLO!\n";

    cudaMallocPitch(&device_input, &pitch, MAX_LEN, N);
    cudaMemcpy2D((void*)device_input,pitch,(void*)strArr,MAX_LEN,MAX_LEN,N,cudaMemcpyHostToDevice);

    cudaMemcpy2D((void*)strArr,MAX_LEN,device_input,pitch,MAX_LEN,N,cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
    char tmpStr[30];
    int i, j, N, pos, range, ret;

    if(argc<5) {
	    cout << "Usage: " << argv[0] << " filename number_of_strings pos range" << endl;
	    return 0;
    }

    ifstream inputfile(argv[1]);

    if(!inputfile.is_open()) {
	    cout << "Unable to open file" << endl;
	    return 0;
    }

    ret=sscanf(argv[2],"%d", &N);
    if(ret==EOF || N<=0) {
	    cout << "Invalid number" << endl;
	    return 0;
    }

    ret=sscanf(argv[3],"%d", &pos);
    if(ret==EOF || pos<0 || pos>=N) {
	    cout << "Invalid position" << endl;
	    return 0;
    }

    ret=sscanf(argv[4],"%d", &range);
    if(ret==EOF || range<0 || (pos+range)>N) {
	    cout << "Invalid range" << endl;
	    return 0;
    }

    auto strArr = new char[N][30];
    memset(strArr,0,sizeof(strArr));

    for(i=0; i<N; i++) inputfile>>strArr[i];

    inputfile.close();

    // Upper Code is the section that get data.
    //radix_sort_cuda(strArr,N);

    cout<<"\nStrings (Names) in Alphabetical order from position " << pos << ": " << endl;
    for(i=pos; i<N && i<(pos+range); i++)
        cout<< i << ": " << strArr[i]<<endl;
    cout<<endl;

    delete[] strArr;

    return 0;
}
