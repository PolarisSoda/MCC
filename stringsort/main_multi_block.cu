#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
#include <iomanip>

using namespace std;

constexpr int MAX_LEN = 32; //String's Max length.
constexpr int CHAR_RANGE = 122 - 64 + 1; //String's char range start with 65 and end with 122. 64 is correspond to null and empty space.
constexpr int NUM_THREADS = 64; //NUM THREAD
constexpr int NUM_BLOCKS = 2; //NUM BLOCKS

__device__ int prefix_offset[NUM_BLOCKS][NUM_THREADS][CHAR_RANGE];

__global__ void kernel_function(char* device_input, char* device_output,char** toggle_index[2], int N) {
    __shared__ int block_histogram[CHAR_RANGE]; //global historam
    __shared__ int block_offset[CHAR_RANGE]; //global offset

    int num_threads = NUM_THREADS * NUM_BLOCKS; //thread의 총 개수.
    int thread_workload = (N+num_threads-1) / num_threads; // thread마다 할당된 block의 양.

    int idx = blockIdx.x*NUM_THREADS + threadIdx.x; //block을 합한 총 thread의 idx
    int local_idx = threadIdx.x;
    int thread_start_pos = idx * thread_workload; //총 arr에서 thread의 시작 위치.
    int thread_end_pos = min(N, thread_start_pos+thread_workload); // thread의 끝 위치.

    int block_workload = (N+NUM_BLOCKS-1) / NUM_BLOCKS; //BLOCK이 처리해야하는 arr양.
    int block_start_pos = blockIdx.x * block_workload; //block의 작업 시작 위치
    int block_end_pos = min(N, block_start_pos+block_workload); //block의 작업 끝 위치.

    for(int i=thread_start_pos; i<thread_end_pos; i++) toggle_index[0][i] = device_input + i*MAX_LEN;

    int input = 0;

    for(int i=thread_start_pos; i<thread_end_pos; i++) {
        for(int j=0; j<MAX_LEN; j++) device_output[i*MAX_LEN + j] = toggle_index[input][i][j];
    }
    __syncthreads();
}

void radix_sort_cuda(char* host_input, char* host_output, int N) {
    size_t data_size = N * MAX_LEN * sizeof(char);

    char* entire_data; //this have the entire data of strings.
    char* output_data; //this is the output.

    cudaMalloc(&entire_data,data_size);
    cudaMalloc(&output_data,data_size);

    cudaMemcpy(entire_data,host_input,data_size,cudaMemcpyHostToDevice);

    char** toggle_index[2];
    cudaMalloc(&toggle_index[0],sizeof(char*)*N);
    cudaMalloc(&toggle_index[1],sizeof(char*)*N);

    kernel_function<<<NUM_BLOCKS,NUM_THREADS>>>(entire_data,output_data,toggle_index,N);

    cudaMemcpy(host_output,output_data,data_size,cudaMemcpyDeviceToHost);

    cudaFree(entire_data);
    cudaFree(output_data);
    cudaFree(toggle_index[0]);
    cudaFree(toggle_index[1]);
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

    auto strArr = new char[N*MAX_LEN];
    auto output = new char[N*MAX_LEN];

    memset(strArr, 64, N * MAX_LEN);
    for (int i=0; i<N; i++) {
        inputfile >> std::setw(MAX_LEN) >> &strArr[i * MAX_LEN];
        int length = strlen(&strArr[i*MAX_LEN]);
        strArr[i*MAX_LEN + length] = 64;
    }
    inputfile.close();

    // Upper Code is the section that get data.
    radix_sort_cuda(strArr,output,N);

    cout << "\nStrings (Names) in Alphabetical order from position " << pos << ": " << "\n";

    for(int i=pos; i<N && i<(pos+range); i++) {
        cout << i << ": ";
        for(int j=0; j<MAX_LEN; j++) {
            char now = output[i*MAX_LEN+j];
            if(now == '@') break;
            cout << now;
        }
        cout << "\n";
    }
    cout << "\n";

    delete[] strArr;
    delete[] output;
    return 0;
}
