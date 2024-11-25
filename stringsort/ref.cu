#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define MAX_LEN 30
#define NUM_BUCKETS 256

__global__ void countSortKernel(char *d_input, char *d_output, int *d_count, int exp, int n, int str_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int j = 0; j < str_len; j++) {
            atomicAdd(&d_count[(d_input[i * str_len + j] / exp) % NUM_BUCKETS], 1);
        }
    }
    __syncthreads();

    if (i < NUM_BUCKETS) {
        for (int j = 1; j < NUM_BUCKETS; j++) {
            d_count[j] += d_count[j - 1];
        }
    }
    __syncthreads();

    if (i < n) {
        for (int j = str_len - 1; j >= 0; j--) {
            int pos = atomicSub(&d_count[(d_input[i * str_len + j] / exp) % NUM_BUCKETS], 1) - 1;
            d_output[pos * str_len + j] = d_input[i * str_len + j];
        }
    }
}

void radixSort(char h_input[][MAX_LEN], int n) {
    char *d_input, *d_output;
    int *d_count;
    size_t size = n * MAX_LEN * sizeof(char);
    size_t count_size = NUM_BUCKETS * sizeof(int);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_count, count_size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    for (int exp = 1; exp < MAX_LEN; exp *= NUM_BUCKETS) {
        cudaMemset(d_count, 0, count_size);
        countSortKernel<<<(n + 255) / 256, 256>>>(d_input, d_output, d_count, exp, n, MAX_LEN);
        cudaMemcpy(d_input, d_output, size, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_input, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
}

int main() {
    char h_input[][MAX_LEN] = {"hi", "hello", "you", "this", "is", "for", "ttttt"};
    int n = sizeof(h_input) / MAX_LEN;

    for(int i=0; i<n; i++) printf("First string: %s\n", h_input[i]);
    
    radixSort(h_input, n);

    for (int i = 0; i < n; i++) {
        printf("Sorted string: %s\n", h_input[i]);
    }

    return 0;
}