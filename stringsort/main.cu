#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>

using namespace std;

constexpr int MAX_LEN = 32;
constexpr int CHAR_RANGE = 122 - 64 + 1;
constexpr int NUM_THREADS = 512;

__global__ void kernel_function(char* device_input, char* device_output, int N, int pos) {
    int idx = threadIdx.x;
    int workload = (N + NUM_THREADS - 1) / NUM_THREADS;

    int start_pos = threadIdx.x * workload;
    int end_pos = min(N, start_pos + workload);

    for (int i = start_pos; i < end_pos; i++) {
        for (int j = 0; j < MAX_LEN; j++) {
            device_output[i * MAX_LEN + j] = device_input[i * MAX_LEN + j];
        }
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

void radix_sort_cuda(char* host_input, char* host_output, int N) {
    size_t data_size = N * MAX_LEN * sizeof(char);

    char* device_input;
    char* device_output;

    checkCudaError(cudaMalloc(&device_input, data_size), "cudaMalloc device_input");
    checkCudaError(cudaMalloc(&device_output, data_size), "cudaMalloc device_output");

    checkCudaError(cudaMemcpy(device_input, host_input, data_size, cudaMemcpyHostToDevice), "cudaMemcpy host to device");

    kernel_function<<<1, NUM_THREADS>>>(device_input, device_output, N, MAX_LEN - 1);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    checkCudaError(cudaMemcpy(host_output, device_output, data_size, cudaMemcpyDeviceToHost), "cudaMemcpy device to host");

    cudaFree(device_input);
    cudaFree(device_output);
}

int main(int argc, char* argv[]) {
    int N, pos, range, ret;

    if (argc < 5) {
        cout << "Usage: " << argv[0] << " filename number_of_strings pos range" << endl;
        return 0;
    }

    ifstream inputfile(argv[1]);

    if (!inputfile.is_open()) {
        cout << "Unable to open file" << endl;
        return 0;
    }

    ret = sscanf(argv[2], "%d", &N);
    if (ret == EOF || N <= 0) {
        cout << "Invalid number" << endl;
        return 0;
    }

    ret = sscanf(argv[3], "%d", &pos);
    if (ret == EOF || pos < 0 || pos >= N) {
        cout << "Invalid position" << endl;
        return 0;
    }

    ret = sscanf(argv[4], "%d", &range);
    if (ret == EOF || range < 0 || (pos + range) > N) {
        cout << "Invalid range" << endl;
        return 0;
    }

    auto strArr = new char[N * MAX_LEN];
    auto output = new char[N * MAX_LEN];

    memset(strArr, 64, N * MAX_LEN);
    for (int i = 0; i < N; i++) {
        char temp_arr[MAX_LEN];
        inputfile >> temp_arr;

        int length = strlen(temp_arr);
        int pos = MAX_LEN - length;

        memcpy(&strArr[i * MAX_LEN + pos], temp_arr, length);
    }
    inputfile.close();

    radix_sort_cuda(strArr, output, N);

    cout << "\nStrings (Names) in Alphabetical order from position " << pos << ": " << "\n";
    for (int i = pos; i < N; i++) {
        cout << i << ": ";
        for (int j = 0; j < MAX_LEN; j++) cout << output[i * MAX_LEN + j];
        cout << endl;
    }

    cout << "\n";

    delete[] strArr;
    delete[] output;

    return 0;
}