#include <cuda_runtime.h>
#include <stdio.h>

__global__ void doubleArrayKernel(int *d_array, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i * cols + j;
    if (i < rows && j < cols) {
        d_array[index] *= 2;
    }
}

void doubleArray(int *h_array, int rows, int cols) {
    int *d_array;
    size_t size = rows * cols * sizeof(int);

    // 메모리 할당
    cudaMalloc((void**)&d_array, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    dim3 blockSize(16, 16);
    dim3 numBlocks((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);
    doubleArrayKernel<<<numBlocks, blockSize>>>(d_array, rows, cols);
    cudaDeviceSynchronize();

    // 결과를 호스트로 복사
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_array);
}

int main() {
    int h_array[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    int rows = 3;
    int cols = 2;

    doubleArray((int*)h_array, rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("Array element [%d][%d]: %d\n", i, j, h_array[i][j]);
        }
    }

    return 0;
}