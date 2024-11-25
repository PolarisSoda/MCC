#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>

using namespace std;

constexpr int MAX_LEN = 32;

__global__ void kernel_function(char* device_input, char* device_output, int N) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < N) {
        // Each thread works on one row
        char* row = device_input + row_idx * MAX_LEN;
        char* sorted_row = device_output + row_idx * MAX_LEN;

        // Perform sorting on this row (simple example of sorting one string)
        for (int i = 0; i < MAX_LEN; ++i) {
            for (int j = i + 1; j < MAX_LEN; ++j) {
                if (row[i] > row[j]) {
                    char temp = row[i];
                    row[i] = row[j];
                    row[j] = temp;
                }
            }
        }

        // Copy the sorted row to the output
        for (int i = 0; i < MAX_LEN; ++i) {
            sorted_row[i] = row[i];
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

    // Now we conduct real radix sort parallel.
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    kernel_function<<<num_blocks,threads_per_block>>>(device_input,device_output,N);

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
    memset(strArr,0,sizeof(strArr));

    for(int i=0; i<N; i++) {
        char temp_arr[MAX_LEN];
        // inputfile>>strArr[i];
        inputfile >> temp_arr;
        if(strlen(temp_arr) >= 30) cout << strlen(temp_arr) << endl;
    }
    inputfile.close();

    // Upper Code is the section that get data.
    radix_sort_cuda(strArr,N);

    cout << "\nStrings (Names) in Alphabetical order from position " << pos << ": " << "\n";
    for(int i=pos; i<N && i<(pos+range); i++) cout << i << ": " << strArr[i] << "\n";
    cout << "\n";

    delete[] strArr;

    return 0;
}
