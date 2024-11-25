#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>

constexpr int MAX_LEN = 30;  // Maximum string length

// Kernel to perform counting sort for a specific character position (char_pos)
__global__ void counting_sort_kernel(char* device_input, char* device_output, int N, int char_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't go out of bounds
    if (idx < N) {
        // Get the string at index 'idx'
        char* input_str = device_input + idx * MAX_LEN;
        char* output_str = device_output + idx * MAX_LEN;

        // Extract the character at the specified position
        char char_at_pos = input_str[char_pos];
        
        // Just copy the character to the output for now
        output_str[char_pos] = char_at_pos;
        
        // Perform a more complex counting sort logic here (this is simplified)
    }
}

// Radix sort function that calls counting_sort_kernel iteratively for each character
void radix_sort_cuda(char strArr[][MAX_LEN], int N) {
    size_t data_size = N * MAX_LEN * sizeof(char);
    char* device_input;
    char* device_output;

    cudaMalloc(&device_input, data_size);
    cudaMalloc(&device_output, data_size);

    // Copy input strings from host to device
    cudaMemcpy(device_input, strArr, data_size, cudaMemcpyHostToDevice);

    // Perform radix sort on each character position (starting from the least significant)
    for (int char_pos = MAX_LEN - 1; char_pos >= 0; --char_pos) {
        int threads_per_block = 256;
        int num_blocks = (N + threads_per_block - 1) / threads_per_block;

        // Launch kernel for sorting strings based on the current character position
        counting_sort_kernel<<<num_blocks, threads_per_block>>>(device_input, device_output, N, char_pos);

        // Swap input and output pointers for the next pass
        cudaMemcpy(device_input, device_output, data_size, cudaMemcpyDeviceToDevice);
    }

    // Copy the final sorted array from device back to host
    cudaMemcpy(strArr, device_input, data_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
}

int main() {
    // Example usage
    const int N = 5;
    char strArr[N][MAX_LEN] = {"apple", "orange", "banana", "grape", "cherry"};
    
    // Call CUDA radix sort function
    radix_sort_cuda(strArr, N);

    // Output the sorted strings
    for (int i = 0; i < N; ++i) {
        printf("%s\n", strArr[i]);
    }

    return 0;
}