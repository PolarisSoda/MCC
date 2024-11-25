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

    // 메모리 할당
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_count, count_size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int exp = 1; exp < NUM_BUCKETS; exp *= NUM_BUCKETS) {
        cudaMemset(d_count, 0, count_size);
        countSortKernel<<<numBlocks, blockSize>>>(d_input, d_output, d_count, exp, n, MAX_LEN);
        cudaDeviceSynchronize();

        // 결과를 d_input으로 복사
        cudaMemcpy(d_input, d_output, size, cudaMemcpyDeviceToDevice);
    }

    // 결과를 호스트로 복사
    cudaMemcpy(h_input, d_output, size, cudaMemcpyDeviceToHost);

    // 메모리 해제
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