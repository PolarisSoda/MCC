#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
#include <iomanip>

using namespace std;

constexpr int MAX_LEN = 32; //String's Max length.
constexpr int CHAR_RANGE = 122 - 64 + 1; //String's char range start with 65 and end with 122. 64 is correspond to null and empty space.
constexpr int NUM_THREADS = 16; //NUM THREAD
constexpr int NUM_BLOCKS = 64; //NUM BLOCKS

__global__ void kernel_function(char* device_input, char* device_output, char** input_index, char** output_index, int N) {
    __shared__ int block_histogram[CHAR_RANGE]; //global historam
    __shared__ int block_offset[CHAR_RANGE]; //global offset
    __shared__ int block_count[CHAR_RANGE]; //global count

    int num_threads = NUM_THREADS * NUM_BLOCKS;
    int thread_workload = (N+num_threads-1) / num_threads;

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int thread_start_pos = idx * thread_workload;
    int thread_end_pos = min(N, thread_start_pos+thread_workload);

    int block_workload = (N +blockDim.x-1) / blockDim.x;
    int block_start_pos = blockIdx.x * block_workload;
    int block_end_pos = min(N, block_start_pos+block_workload);

    for(int i=thread_start_pos; i<thread_end_pos; i++) input_index[i] = device_input + i*MAX_LEN;

    for(int pos=MAX_LEN-1; pos>=0; pos--) {
        //INIT GLOBAL VARIABLE
        if(threadIdx.x < CHAR_RANGE) block_histogram[threadIdx.x] = 0, block_count[threadIdx.x] = 0;
        __syncthreads();

        int local_histogram[CHAR_RANGE] = {0,};
        for(int i=thread_start_pos; i<thread_end_pos; i++) {
            char now = input_index[i][pos];
            local_histogram[now-64]++;
        }
        for(int i=0; i<CHAR_RANGE; i++) atomicAdd(&block_histogram[i],local_histogram[i]);
        __syncthreads();

        if(idx == 0) {
            block_offset[0] = 0;
            for(int i=0; i<CHAR_RANGE-1; i++) block_offset[i+1] = block_offset[i] + block_histogram[i];
        }
        __syncthreads();

        for(int i=block_start_pos; i<block_end_pos; i++) {
            char now = input_index[i][pos];
            int index = now - 64;

            if(threadIdx.x == index) {
                int after_index = block_start_pos + block_offset[index] + block_count[index]++;
                output_index[after_index] = input_index[i];
            }
        }
        __syncthreads();

        for(int i=thread_start_pos; i<thread_end_pos; i++) {
            char* swap_temp = input_index[i];
            input_index[i] = output_index[i];
            output_index[i] = swap_temp;

        }
        __syncthreads();
    }

    for(int i=thread_start_pos; i<thread_end_pos; i++) {
        for(int j=0; j<MAX_LEN; j++) device_output[i*MAX_LEN + j] = input_index[i][j];
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

    char** input_index;
    char** output_index;

    cudaMalloc(&input_index,sizeof(char*)*N);
    cudaMalloc(&output_index,sizeof(char*)*N);

    kernel_function<<<NUM_BLOCKS,NUM_THREADS>>>(entire_data,output_data,input_index,output_index,N);

    cudaMemcpy(host_output,output_data,data_size,cudaMemcpyDeviceToHost);

    cudaFree(entire_data);
    cudaFree(output_data);
    cudaFree(input_index);
    cudaFree(output_index);
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
