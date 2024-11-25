#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>

using namespace std;

constexpr int MAX_LEN = 32;
constexpr int CHAR_RANGE = 122 - 64 + 1;
constexpr int NUM_THREADS = 512;
//65 ~ 122

__global__ void kernel_function(char* device_input, char* device_output, int N, int pos) {
    //N is total amount of string.
    //we have NUM_THREADS 512
    //each THREAD HAVE 196 strings.

    __shared__ int histogram[CHAR_RANGE];
    __shared__ int offset[CHAR_RANGE];
    __shared__ int count[CHAR_RANGE];


    int idx = threadIdx.x;
    int workload = (N + NUM_THREADS - 1) / NUM_THREADS; //각 스레드가 가지는 문자열의 양.

    int start_pos = threadIdx.x * 196; // 0: 0~195 1: 196~391 //각 스레드가 시작할 위치.

    int end_pos = min(N,start_pos + workload);

    if (idx < CHAR_RANGE) {
        histogram[idx] = 0;
        count[idx] = 0;
    }
    __syncthreads();

    //out char value is 64 ~ 123, 64 is for null values.
    for (int i = start_pos; i < end_pos; i++) {
        char now = device_input[i * MAX_LEN + pos];
        atomicAdd(&histogram[now-64], 1);
    }
    __syncthreads();

    if(idx == 0) {
        offset[0] = 0;

        for(int i=0; i<CHAR_RANGE-1; i++) {
            offset[i+1] = offset[i] + histogram[i];
        }
    }
    __syncthreads();
    
    for (int i = start_pos; i < end_pos; i++) {
        for (int j=0; j <MAX_LEN; j++) {
            device_output[i * MAX_LEN + j] = device_input[i * MAX_LEN + j];
        }
    }
}


void radix_sort_cuda(char strArr[][MAX_LEN], int N) {

    // First we have to copy these data to device.
    size_t data_size = N * MAX_LEN * sizeof(char);

    char* device_input;
    char* device_output;

    cudaMalloc(&device_input, data_size);
    cudaMalloc(&device_output, data_size);

    cudaMemcpy(device_input, strArr, data_size, cudaMemcpyHostToDevice);

    kernel_function<<<1,NUM_THREADS>>>(device_input,device_output,N,MAX_LEN-1);

    // and we give output to host.
    cudaMemcpy(strArr,device_output,data_size,cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
    int N, pos, range, ret;

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

    auto strArr = new char[N][MAX_LEN];
    auto outputs = new char[N][MAX_LEN];

    for(int i = 0; i < N; i++) {
        char temp_arr[MAX_LEN];
        inputfile >> temp_arr;

        int length = strlen(temp_arr);
        int pos = MAX_LEN - length;

        memset(strArr[i],64,MAX_LEN);
        memcpy(&strArr[i][pos], temp_arr, length);
    }
    inputfile.close();

    for(int i=0; i<N; i++) {
        for(int j=0; j<MAX_LEN; j++) {
            if(strArr[i][j] == 0) cout << 0;
            else cout << strArr[i][j];
        }
        cout << endl;
    }

    // Upper Code is the section that get data.
    radix_sort_cuda(strArr,N);

    cout << "\nStrings (Names) in Alphabetical order from position " << pos << ": " << "\n";
    for(int i=pos; i<N && i<(pos+range); i++) {
        cout << i << ": ";
        for(int j=0; j<MAX_LEN; j++) cout << strArr[i][j];
        cout << endl;
    }
        
    cout << "\n";

    delete[] strArr;

    return 0;
}
