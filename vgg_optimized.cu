/* Includes, system */
#include <cstdio>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

struct Tensor4d
{
    float *data;
    size_t data_size;

    Tensor4d(int n, int c, int h, int w)
    {
        data_size = n * c * h * w;
        cudaMalloc((void **)&data, data_size * sizeof(float));
    }

    ~Tensor4d()
    {
        cudaFree(data);
    }
};




struct Bias4d
{
    float *data;
    size_t data_size;

    Bias4d(int n, int c, int h, int w)
    {
        data_size = n * c * h * w;
        float *tmp = (float *)malloc(data_size * sizeof(float));
        for (int i = 0; i < data_size; i++)
        {
            tmp[i] = (float)std::rand() / RAND_MAX / 1000;
        }
        cudaMalloc((void **)&data, data_size * sizeof(float));
        cudaMemcpy(data, tmp, data_size * sizeof(float), cudaMemcpyHostToDevice);
        free(tmp);
    }

    ~Bias4d()
    {
        cudaFree(data);
    }
};




struct Filter4d
{
    float *data;
    size_t data_size;

    Filter4d(int n, int c, int h, int w)
    {
        data_size = n * c * h * w;
        float *tmp = (float *)malloc(data_size * sizeof(float));
        for (int i = 0; i < data_size; i++)
        {
            tmp[i] = (float)std::rand() / RAND_MAX / 1000;
        }
        cudaMalloc((void **)&data, data_size * sizeof(float));
        cudaMemcpy(data, tmp, data_size * sizeof(float), cudaMemcpyHostToDevice);
        free(tmp);
    }

    ~Filter4d()
    {
        cudaFree(data);
    }
};




__global__ void reluActivation(float *input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size)
    {
        input[idx] = fmaxf(input[idx], 0.0f);
    }
}




void cuReLU(float *input, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    reluActivation<<<numBlocks, blockSize>>>(input, size);
    cudaDeviceSynchronize();
}



__global__ void convolution2d(float *input, float *filter, float *output,
                              int input_width, int input_height, int input_channels,
                              int filter_width, int filter_height,
                              int output_width, int output_height,
                              int pad_w, int pad_h,
                              int wstride, int hstride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if (idx < output_width && idy < output_height && idz < input_channels)
    {
        int out_idx = idz * output_width * output_height + idy * output_width + idx;

        float value = 0.0f;

        for (int fy = 0; fy < filter_height; ++fy)
        {
            int in_y = idy * hstride + fy - pad_h;
            for (int fx = 0; fx < filter_width; ++fx)
            {
                int in_x = idx * wstride + fx - pad_w;
                if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height)
                {
                    int in_idx = idz * input_width * input_height + in_y * input_width + in_x;
                    int filter_idx = idz * filter_width * filter_height + fy * filter_width + fx;
                    value += input[in_idx] * filter[filter_idx];
                }
            }
        }

        output[out_idx] = value;
    }
}



__global__ void addBias(float *output, float *bias, int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size)
    {
        output[idx] += bias[idx];
    }
}




void cuConv2D(float *input, float *output, int w, int h, int c, int n, int k,
              int filter_w, int filter_h, int dilation_w, int dilation_h,
              int pad_w, int pad_h, int wstride, int hstride)
{
    // Allocate device memory for input
    float *d_output;

    // Compute output dimensions
    int out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) / hstride + 1;
    int out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) / wstride + 1;

    // Allocate device memory for output
    cudaMalloc((void **)&d_output, n * k * out_h * out_w * sizeof(float));

    // Create filter and bias
    Filter4d w_desc(k, c, filter_w, filter_h);
    Bias4d bias(k, 1, 1, 1); // bias is 1x1x1xk

    // Compute grid and block dimensions for convolution kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   c);

    // Perform convolution
    auto start = std::chrono::steady_clock::now();
    convolution2d<<<numBlocks, threadsPerBlock>>>(input, w_desc.data, d_output,
                                                  w, h, c, filter_w, filter_h,
                                                  out_w, out_h, pad_w, pad_h,
                                                  wstride, hstride);

    // Add bias
    addBias<<<(n * k + 255) / 256, 256>>>(d_output, bias.data, n * k * out_h * out_w);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                    std::micro>(end - start).count());

    std::cout << " " << fwd_time << " us" << std::endl;

    // Copy output data to the output parameter
    cudaMemcpy(output, d_output, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free device memory
    cudaFree(d_output);
}





__global__ void maxPoolingKernel(float* input, float* output, int width, int height, int channels, int poolSizeW, int poolSizeH, int strideW, int strideH) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z;

    if (outX < (width / strideW) && outY < (height / strideH)) {
        float maxVal = -FLT_MAX;
        int inXOrigin = outX * strideW;
        int inYOrigin = outY * strideH;
        for (int dy = 0; dy < poolSizeH; ++dy) {
            for (int dx = 0; dx < poolSizeW; ++dx) {
                int inX = inXOrigin + dx;
                int inY = inYOrigin + dy;
                if (inX < width && inY < height) {
                    int idx = (outZ * height + inY) * width + inX;
                    float val = input[idx];
                    if (val > maxVal) {
                        maxVal = val;
                    }
                }
            }
        }
        int outIdx = (outZ * (height / strideH) + outY) * (width / strideW) + outX;
        output[outIdx] = maxVal;
    }
}





void cuMaxPool(float *input, float *output, int w, int h, int c, int n) {
    // Assume pool size and strides
    int poolSizeW = 2;
    int poolSizeH = 2;
    int strideW = 2;
    int strideH = 2;

    // Calculate output dimensions
    int outW = w / strideW;
    int outH = h / strideH;

    // Allocate memory on device
    float *d_output;
    cudaMalloc(&d_output, n * c * outH * outW * sizeof(float));

    // Setup grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((outW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (outH + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   c * n);

    // Launch kernel
    auto start = std::chrono::steady_clock::now();
    maxPoolingKernel<<<numBlocks, threadsPerBlock>>>(input, d_output, w, h, c, poolSizeW, poolSizeH, strideW, strideH);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    std::cout << "Kernel Execution Time: " << elapsed.count() << " us" << std::endl;

    // Copy output data back to output parameter
    cudaMemcpy(output, d_output, n * c * outH * outW * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free device memory
    cudaFree(d_output);
}



template<int TILE_WIDTH>

__global__ void matmul_shared_mem(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    const unsigned int cRow = blockIdx.x;
    const unsigned int cCol = blockIdx.y;

    __shared__ float As[TILE_WIDTH * TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH * TILE_WIDTH];

    const unsigned int threadCol = threadIdx.x % TILE_WIDTH;
    const unsigned int threadRow = threadIdx.x / TILE_WIDTH;

    if(threadRow<M && threadCol<N) {
        A += cRow * TILE_WIDTH * K;
        B += cCol * TILE_WIDTH;
        C += cRow * TILE_WIDTH * N + cCol * TILE_WIDTH;

        float tmp = 0.0;
        for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_WIDTH) {
            As[threadRow * TILE_WIDTH + threadCol] = A[threadRow * K + threadCol];
            Bs[threadRow * TILE_WIDTH + threadCol] = B[threadRow * N + threadCol];

            __syncthreads();
            A += TILE_WIDTH;
            B += TILE_WIDTH * N;

            for (int dotIdx = 0; dotIdx < TILE_WIDTH; ++dotIdx) {
                tmp += As[threadRow * TILE_WIDTH + dotIdx] *
                       Bs[dotIdx * TILE_WIDTH + threadCol];
            }
            __syncthreads();
        }
        C[threadRow * N + threadCol] =
            alpha * tmp + beta * C[threadRow * N + threadCol];
    }
}


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cuFC(float *input, float *output, int left, int right)
{
    int lda = 1, ldb = left, ldc = 1, m = 1, k = left, n = right;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    float *h_B = (float *)malloc(left * right * sizeof(float));
    for (int i = 0; i < left * right; i++)
    {
        h_B[i] = (float)std::rand() / RAND_MAX / 1000;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, left * sizeof(float));
    cudaMalloc(&d_B, left * right * sizeof(float));
    cudaMalloc(&d_C, right * sizeof(float));

    cudaMemcpy(d_A, input, left * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, left * right * sizeof(float), cudaMemcpyHostToDevice);



    auto start = std::chrono::steady_clock::now();

    dim3 gridDim(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    dim3 blockDim(32 * 32);
    matmul_shared_mem<32><<<gridDim, blockDim>>>(m, n, k, 1.0, d_A, d_B, 0.0, d_C);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                    std::micro>(end - start).count());

    std::cout << " " << fwd_time << " ms" << std::endl;

    cudaMemcpy(output, d_C, right * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



int main()
{
    std::srand(std::time(0));

    float *input;

    int data_size = 224 * 224 * 3 * 1;
    input = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; i++)
    {
        input[i] = (float)std::rand() / RAND_MAX;
    }

    // Allocate device memory for input
    float *d_input, *d_output;
    cudaMalloc(&d_input, data_size * sizeof(float));
    cudaMemcpy(d_input, input, data_size * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();

    // Block 1
    std::cout << "CONV 224x224x64";
    cudaMalloc(&d_output, 224 * 224 * 64 * sizeof(float));
    cuConv2D(d_input, d_output, 224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 224 * 224 * 64);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "CONV 224x224x64";
    cudaMalloc(&d_output, 224 * 224 * 64 * sizeof(float));
    cuConv2D(d_input, d_output, 224, 224, 64, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 224 * 224 * 64);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "POOLMAX 112x112x64";
    cudaMalloc(&d_output, 112 * 112 * 64 * sizeof(float));
    cuMaxPool(d_input, d_output, 224, 224, 64, 1);
    cudaFree(d_input);
    d_input = d_output;

    // Block 2
    std::cout << "CONV 112x112x128";
    cudaMalloc(&d_output, 112 * 112 * 128 * sizeof(float));
    cuConv2D(d_input, d_output, 112, 112, 64, 1, 128, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 112 * 112 * 128);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "CONV 112x112x128";
    cudaMalloc(&d_output, 112 * 112 * 128 * sizeof(float));
    cuConv2D(d_input, d_output, 112, 112, 128, 1, 128, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 112 * 112 * 128);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "POOLMAX 56x56x128";
    cudaMalloc(&d_output, 56 * 56 * 128 * sizeof(float));
    cuMaxPool(d_input, d_output, 112, 112, 128, 1);
    cudaFree(d_input);
    d_input = d_output;

    // Block 3
    std::cout << "CONV 56x56x256";
    cudaMalloc(&d_output, 56 * 56 * 256 * sizeof(float));
    cuConv2D(d_input, d_output, 56, 56, 128, 1, 256, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 56 * 56 * 256);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "CONV 56x56x256";
    cudaMalloc(&d_output, 56 * 56 * 256 * sizeof(float));
    cuConv2D(d_input, d_output, 56, 56, 256, 1, 256, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 56 * 56 * 256);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "POOLMAX 28x28x256";
    cudaMalloc(&d_output, 28 * 28 * 256 * sizeof(float));
    cuMaxPool(d_input, d_output, 56, 56, 256, 1);
    cudaFree(d_input);
    d_input = d_output;

    // Block 4
    std::cout << "CONV 28x28x512";
    cudaMalloc(&d_output, 28 * 28 * 512 * sizeof(float));
    cuConv2D(d_input, d_output, 28, 28, 256, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 28 * 28 * 512);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "CONV 28x28x512";
    cudaMalloc(&d_output, 28 * 28 * 512 * sizeof(float));
    cuConv2D(d_input, d_output, 28, 28, 512, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 28 * 28 * 512);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "POOLMAX 14x14x512";
    cudaMalloc(&d_output, 14 * 14 * 512 * sizeof(float));
    cuMaxPool(d_input, d_output, 28, 28, 512, 1);
    cudaFree(d_input);
    d_input = d_output;

    // Block 5
    std::cout << "CONV 14x14x512";
    cudaMalloc(&d_output, 14 * 14 * 512 * sizeof(float));
    cuConv2D(d_input, d_output, 14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 14 * 14 * 512);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "CONV 14x14x512";
    cudaMalloc(&d_output, 14 * 14 * 512 * sizeof(float));
    cuConv2D(d_input, d_output, 14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(d_output, 14 * 14 * 512);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "POOLMAX 7x7x512";
    cudaMalloc(&d_output, 7 * 7 * 512 * sizeof(float));
    cuMaxPool(d_input, d_output, 14, 14, 512, 1);
    cudaFree(d_input);
    d_input = d_output;

    // Fully connected layers
    std::cout << "FC 4096";
    cudaMalloc(&d_output, 4096 * sizeof(float));
    cuFC(d_input, d_output, 7 * 7 * 512, 4096);
    cuReLU(d_output, 4096);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "FC 4096";
    cudaMalloc(&d_output, 4096 * sizeof(float));
    cuFC(d_input, d_output, 4096, 4096);
    cuReLU(d_output, 4096);
    cudaFree(d_input);
    d_input = d_output;

    std::cout << "FC 1000";
    cudaMalloc(&d_output, 1000 * sizeof(float));
    cuFC(d_input, d_output, 4096, 1000);

    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                                          std::micro>(end - start)
                                      .count());

    std::cout << "total time " << fwd_time << " us" << std::endl;

    // Copy final output to host
    float *final_output = (float *)malloc(1000 * sizeof(float));
    cudaMemcpy(final_output, d_output, 1000 * sizeof(float), cudaMemcpyDeviceToHost);

    free(input);
    free(final_output);
    cudaFree(d_output);

    return 0;
}
