#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>

using namespace std;

constexpr int MAX_LEN = 32;

__global__ void kernel_function(char* device_input, char* device_output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        char* input_str = device_input + idx * MAX_LEN;
        char* output_str = device_output + idx * MAX_LEN;

        // 문자열을 뒤집고 null 문자를 포함하여 처리
        for (int i = 0; i < MAX_LEN; ++i) {
            output_str[MAX_LEN-1-i] = input_str[i];
        }
        output_str[MAX_LEN] = '\0'; // null 문자 추가
    }
}
void radix_sort_cuda(char strArr[][MAX_LEN], int N, char output[][MAX_LEN]) {

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
    cudaMemcpy(output,device_output,data_size,cudaMemcpyDeviceToHost);
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

    auto strArr = new char[N][32];
    auto outputs = new char[N][32];
    memset(strArr,0,sizeof(strArr));

    for(int i=0; i<N; i++) inputfile>>strArr[i];

    inputfile.close();

    // Upper Code is the section that get data.
    radix_sort_cuda(strArr,N,outputs);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < MAX_LEN / 2; j++) {
            swap(outputs[i][j], outputs[i][MAX_LEN-1-j]);
        }
        outputs[i][MAX_LEN] = '\0'; // null 문자 추가
        int checker = strcmp(strArr[i], outputs[i]);
        if (checker != 0) {
            cout << strArr[i] << " " << outputs[i] << endl;
            cout << i << endl;
        }
    }

    exit(0);
    cout << "\nStrings (Names) in Alphabetical order from position " << pos << ": " << "\n";
    for(int i=pos; i<N && i<(pos+range); i++)
        cout<< i << ": " << strArr[i] << "\n";
    cout << "\n";

    delete[] strArr;

    return 0;
}
