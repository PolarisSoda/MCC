#include <cuda_runtime.h>
#include <stdio.h>

__global__ void doubleArrayKernel(int *d_array, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_array[i] *= 2;
    }
}

void doubleArray(int *h_array, int n) {
    int *d_array;
    size_t size = n * sizeof(int);

    // 메모리 할당
    cudaMalloc((void**)&d_array, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    doubleArrayKernel<<<numBlocks, blockSize>>>(d_array, n);
    cudaDeviceSynchronize();

    // 결과를 호스트로 복사
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_array);
}

int main() {
    int h_array[] = {1, 2, 3, 4, 5};
    int n = sizeof(h_array) / sizeof(h_array[0]);

    doubleArray(h_array, n);

    for (int i = 0; i < n; i++) {
        printf("Array element %d: %d\n", i, h_array[i]);
    }

    return 0;
}