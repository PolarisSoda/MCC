#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
#include <iomanip>
#include <assert.h>

using namespace std;

constexpr int MAX_LEN = 32; //String's Max length.
constexpr int CHAR_RANGE = 122 - 64 + 1; //String's char range start with 65 and end with 122. 64 is correspond to null and empty space.
constexpr int NUM_THREADS = 64; //NUM THREAD
constexpr int NUM_BLOCKS = 32; //NUM BLOCKS

__constant__ char MAX_INF_STR[] = "~";
__constant__ char MIN_INF_STR[] = "0";

__device__ int device_strncmp(const char* a, const char* b, int n) {
    for(int i = 0; i < n; ++i){
        if(a[i] < b[i]) return -1;
        if(a[i] > b[i]) return 1;
        if(a[i] == '\0') return 0;
    }
    return 0;
}

__global__ void kernel_function(char* device_input, char* device_output, char** input_index, char** output_index, int N) {
    __shared__ int block_histogram[CHAR_RANGE]; //global historam
    __shared__ int block_offset[CHAR_RANGE]; //global offset
    __shared__ int prefix_offset[NUM_THREADS][CHAR_RANGE];

    int num_threads = NUM_THREADS * NUM_BLOCKS; //thread의 총 개수.
    int thread_workload = (N+num_threads-1) / num_threads; // thread마다 할당된 block의 양.

    int idx = blockIdx.x*blockDim.x + threadIdx.x; //block을 합한 총 thread의 idx
    int local_idx = threadIdx.x;

    int thread_start_pos = idx * thread_workload; //총 arr에서 thread의 시작 위치.
    int thread_end_pos = min(N, thread_start_pos+thread_workload); // thread의 끝 위치.

    int block_workload = thread_workload*NUM_THREADS;
    int block_start_pos = blockIdx.x * block_workload; //block의 작업 시작 위치
    int block_end_pos = min(N, block_start_pos + block_workload); //block의 작업 끝 위치.

    for(int i=thread_start_pos; i<thread_end_pos; i++) input_index[i] = device_input + i*MAX_LEN;

    for(int pos=MAX_LEN-1; pos>=0; pos--) {
        // INIT global variable
        if(local_idx < CHAR_RANGE) block_histogram[local_idx] = 0;
        for(int i=0; i<CHAR_RANGE; i++) prefix_offset[local_idx][i] = 0;
        __syncthreads();

        int local_histogram[CHAR_RANGE] = {0,};
        for(int i=thread_start_pos; i<thread_end_pos; i++) {
            char now = input_index[i][pos];
            local_histogram[now-64]++;
        }

        for(int i=0; i<CHAR_RANGE; i++) {
            atomicAdd(&block_histogram[i],local_histogram[i]);
            prefix_offset[local_idx][i] = local_histogram[i];
        }
        __syncthreads();

        // 이거 얼마 안걸린다
        int prefix_count[CHAR_RANGE] = {0,};
        for(int i=0; i<local_idx; i++) {
            for(int j=0; j<CHAR_RANGE; j++) prefix_count[j] += prefix_offset[i][j];
        }

        if(local_idx == 0) {
            block_offset[0] = 0;
            for(int i=0; i<CHAR_RANGE-1; i++) block_offset[i+1] = block_offset[i] + block_histogram[i];
        }
        __syncthreads();

        int local_count[CHAR_RANGE] = {0,};
        for(int i=thread_start_pos; i<thread_end_pos; i++) {
            char now = input_index[i][pos];
            int index = now - 64;
            int after_index = block_start_pos + block_offset[index] + prefix_count[index] + local_count[index]++;
            output_index[after_index] = input_index[i];
        }
        __syncthreads();
        for(int i=thread_start_pos; i<thread_end_pos; i++) input_index[i] = output_index[i];
    }
}

__global__ void kernel_merge(char* device_input, char* device_output, char** input_index, char** output_index, int N) {
    __shared__ int start_pos[NUM_BLOCKS];
    __shared__ int end_pos[NUM_BLOCKS];
    

    int num_threads = NUM_THREADS * NUM_BLOCKS; //thread의 총 개수.
    int thread_workload = (N+num_threads-1) / num_threads; // thread마다 할당된 block의 양.
    int idx = threadIdx.x; //block안에서의 thread.
    int block_workload = thread_workload*NUM_THREADS;

    start_pos[idx] = idx * block_workload;
    end_pos[idx] = min(N, start_pos[idx]+block_workload);
    
    for(int i=0; i<1; i++) {
        if(i%2 == 0) {
            if(idx % 2 == 0) {
                int left_cur = start_pos[idx];
                int left_end = end_pos[idx];
                int right_cur = start_pos[idx+1];
                int right_end = end_pos[idx+1];
                int write_cur = start_pos[idx];

                while(left_cur < left_end && right_cur < right_end) {
                    char* left_str = left_cur == left_end ? MAX_INF_STR : input_index[left_cur];
                    char* right_str = right_cur == right_end ? MAX_INF_STR : input_index[right_cur];
                    int diff = device_strncmp(left_str,right_str,32);

                    if(diff >= 0) output_index[write_cur++] = input_index[right_cur++];
                    else output_index[write_cur++] = input_index[left_cur++];
                }
            }
        } else {
            if(idx % 2 == 1 && idx != NUM_BLOCKS - 1) {
                int write_cur = start_pos[idx];
                int left_cur = start_pos[idx];
                int left_end = end_pos[idx];
                int right_cur = start_pos[idx+1];
                int right_end = end_pos[idx+1];

                while(left_cur < left_end && right_cur < right_end) {
                    char* left_str = left_cur == left_end ? MAX_INF_STR : input_index[left_cur];
                    char* right_str = right_cur == right_end ? MAX_INF_STR : input_index[right_cur];
                    int diff = device_strncmp(left_str,right_str,32);

                    if(diff >= 0) output_index[write_cur++] = input_index[right_cur++];
                    else output_index[write_cur++] = output_index[left_cur++];
                }
            } else if(idx % 2 == 0 && idx != 0) {
                int write_cur = end_pos[idx+1] - 1;
                int left_cur = end_pos[idx] - 1;
                int left_end = start_pos[idx] - 1;
                int right_cur = end_pos[idx+1] - 1;
                int right_end = start_pos[idx+1] - 1;

                while(left_cur > left_end && right_cur > right_end) {
                    char* left_str = left_cur == left_end ? MIN_INF_STR : input_index[left_cur];
                    char* right_str = right_cur == right_end ? MIN_INF_STR : input_index[right_cur];
                    int diff = device_strncmp(left_str,right_str,32);

                    if(diff >= 0) output_index[write_cur--] = input_index[left_cur--];
                    else output_index[write_cur--] = output_index[right_cur--];
                }
            }
        }
        __syncthreads();

        char** swap_temp = input_index;
        input_index = output_index;
        output_index = swap_temp;
        __syncthreads();
    }

    for(int i=start_pos[idx]; i<end_pos[idx]; i++)
        for(int j=0; j<MAX_LEN; j++) device_output[i*MAX_LEN + j] = input_index[i][j];
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
    cudaDeviceSynchronize();

    kernel_merge<<<1,NUM_BLOCKS>>>(entire_data,output_data,input_index,output_index,N);
    cudaDeviceSynchronize();

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
