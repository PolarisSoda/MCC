#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>

using namespace std;

constexpr int MAX_LEN = 32;
constexpr int CHAR_RANGE = 122 - 64 + 1;
constexpr int NUM_THREADS = 512;

__global__ void kernel_function(char* device_input, char* device_output, int N, int pos) {
    __shared__ int histogram[CHAR_RANGE];
    __shared__ int offset[CHAR_RANGE];
    __shared__ int count[CHAR_RANGE];

    int idx = threadIdx.x;
    int workload = (N + NUM_THREADS - 1) / NUM_THREADS;

    int start_pos = threadIdx.x * 196;
    int end_pos = min(N, start_pos + workload);

    if (idx < CHAR_RANGE) {
        histogram[idx] = 0;
        count[idx] = 0;
    }
    __syncthreads();

    for (int i = start_pos; i < end_pos; i++) {
        char now = device_input[i * MAX_LEN + pos];
        atomicAdd(&histogram[now - 64], 1);
    }
    __syncthreads();

    if (idx == 0) {
        offset[0] = 0;
        for (int i = 0; i < CHAR_RANGE - 1; i++) {
            offset[i + 1] = offset[i] + histogram[i];
        }
    }
    __syncthreads();

    for (int i = start_pos; i < end_pos; i++) {
        char now = device_input[i * MAX_LEN + pos];
        int index = now - 64;
        int pos_in_output = offset[index] + atomicAdd(&count[index], 1);
        for (int j = 0; j < MAX_LEN; j++) {
            device_output[pos_in_output * MAX_LEN + j] = device_input[i * MAX_LEN + j];
        }
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << msg << ": " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

void radix_sort_cuda(char strArr[][MAX_LEN], char outputs[][MAX_LEN], int N) {
    size_t data_size = N * MAX_LEN * sizeof(char);

    char* device_input;
    char* device_output;

    checkCudaError(cudaMalloc(&device_input, data_size), "Failed to allocate device_input");
    checkCudaError(cudaMalloc(&device_output, data_size), "Failed to allocate device_output");

    checkCudaError(cudaMemcpy(device_input, strArr, data_size, cudaMemcpyHostToDevice), "Failed to copy to device_input");

    kernel_function<<<1, NUM_THREADS>>>(device_input, device_output, N, MAX_LEN - 1);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    checkCudaError(cudaMemcpy(outputs, device_output, data_size, cudaMemcpyDeviceToHost), "Failed to copy to outputs");

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

    auto strArr = new char[N][MAX_LEN];
    auto outputs = new char[N][MAX_LEN];
    memset(strArr, 0, N * MAX_LEN * sizeof(char));
    memset(outputs, 0, N * MAX_LEN * sizeof(char));

    for (int i = 0; i < N; i++) {
        char temp_arr[MAX_LEN];
        inputfile >> temp_arr;

        int length = strlen(temp_arr);
        int pos = MAX_LEN - length;

        memset(strArr[i], 64, MAX_LEN);
        memcpy(&strArr[i][pos], temp_arr, length);
    }
    inputfile.close();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < MAX_LEN; j++) {
            if (strArr[i][j] == 0) cout << 0;
            else cout << strArr[i][j];
        }
        cout << endl;
    }

    radix_sort_cuda(strArr, outputs, N);

    cout << "\nStrings (Names) in Alphabetical order from position " << pos << ": " << "\n";
    for (int i = pos; i < N && i < (pos + range); i++) {
        cout << i << ": ";
        for (int j = 0; j < MAX_LEN; j++) cout << outputs[i][j];
        cout << endl;
    }

    cout << "\n";

    delete[] strArr;
    delete[] outputs;

    return 0;
}