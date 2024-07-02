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



struct zeros
{
    float *data;
    size_t data_size;

    zeros(std::vector<int> dims)
    {
        data_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        std::vector<float> host_data(data_size, 0.0f);

        cudaMalloc((void **)&data, data_size * sizeof(float));
        cudaMemcpy(data, host_data.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~zeros()
    {
        cudaFree(data);
    }
};



__global__ void reluActivation(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size)
    {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}


void cuReLU(float *input, float *output, int size)
{
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    reluActivation<<<numBlocks, blockSize>>>(d_input, d_output, size);
    
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
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
    // Allocate device memory for input and output
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&d_output, n * k * h * w * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);

    // Create filter and bias
    Filter4d w_desc(k, c, filter_w, filter_h);
    Bias4d bias(k, 1, 1, 1); // bias is 1x1x1xk

    // Compute output dimensions
    int out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) / hstride + 1;
    int out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) / wstride + 1;

    // Allocate device memory for output
    Tensor4d h_desc(n, k, out_h, out_w);

    // Compute grid and block dimensions for convolution kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   c);

    // Perform convolution
    auto start = std::chrono::steady_clock::now();
    convolution2d<<<numBlocks, threadsPerBlock>>>(d_input, w_desc.data, h_desc.data,
                                                  w, h, c, filter_w, filter_h,
                                                  out_w, out_h, pad_w, pad_h,
                                                  wstride, hstride);

    // Add bias
    addBias<<<(n * k + 255) / 256, 256>>>(h_desc.data, bias.data, n * k);

    

    cudaDeviceSynchronize();


    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                    std::micro>(end - start).count());

    std::cout << " " << fwd_time << " ms" << std::endl;
    // Copy output data to host
    cudaMemcpy(output, h_desc.data, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


__global__ void maxPool2d(float *input, float *output,
                          int input_width, int input_height, int input_channels,
                          int pool_width, int pool_height,
                          int output_width, int output_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if (idx < output_width && idy < output_height && idz < input_channels)
    {
        int out_idx = idz * output_width * output_height + idy * output_width + idx;

        int start_x = idx * pool_width;
        int start_y = idy * pool_height;

        float max_val = -INFINITY;

        for (int fy = 0; fy < pool_height; ++fy)
        {
            int in_y = start_y + fy;
            for (int fx = 0; fx < pool_width; ++fx)
            {
                int in_x = start_x + fx;
                if (in_x < input_width && in_y < input_height)
                {
                    int in_idx = idz * input_width * input_height + in_y * input_width + in_x;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }

        output[out_idx] = max_val;
    }
}



void cuMaxPool(float *input, float *output, int w, int h, int c, int n)
{
    // Allocate device memory for input and output
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&d_output, n * c * (h / 2) * (w / 2) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);

    // Compute output dimensions
    int out_h = h / 2;
    int out_w = w / 2;

    // Allocate device memory for output
    Tensor4d h_desc(n, c, out_h, out_w);

    // Compute grid and block dimensions for pooling kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   c);


    // Perform max pooling

    auto start = std::chrono::steady_clock::now();
    maxPool2d<<<numBlocks, threadsPerBlock>>>(d_input, h_desc.data,
                                              w, h, c, 2, 2, out_w, out_h);

    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                    std::micro>(end - start).count());

    std::cout << " " << fwd_time << " ms" << std::endl;


    // Copy output data to host
    cudaMemcpy(output, h_desc.data, n * c * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


__global__ void matrixVectorMultiply(float *A, float *B, float *C, int left, int right)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < right)
    {
        float sum = 0.0f;
        for (int i = 0; i < left; ++i)
        {
            sum += A[i] * B[i * right + idx];
        }
        C[idx] = sum;
    }
}
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// void cuFC(float *input, float *output, int left, int right)
// {
//     // Allocate device memory for input, weights, and output
//     float *d_input, *d_weights, *d_output;
//     cudaMalloc(&d_input, left * sizeof(float));
//     cudaMalloc(&d_weights, left * right * sizeof(float));
//     cudaMalloc(&d_output, right * sizeof(float));

//     // Copy input and weights from host to device
//     cudaMemcpy(d_input, input, left * sizeof(float), cudaMemcpyHostToDevice);

//     float *h_weights = (float *)malloc(left * right * sizeof(float));
//     for (int i = 0; i < left * right; i++)
//     {
//         h_weights[i] = (float)std::rand() / RAND_MAX / 1000;
//     }
//     cudaMemcpy(d_weights, h_weights, left * right * sizeof(float), cudaMemcpyHostToDevice);

//     // Compute grid and block dimensions
//     dim3 threadsPerBlock(256);
//     dim3 numBlocks((right + threadsPerBlock.x - 1) / threadsPerBlock.x);

//     // Perform matrix-vector multiplication
//     auto start = std::chrono::steady_clock::now();
//     matrixVectorMultiply<<<numBlocks, threadsPerBlock>>>(d_weights, d_input, d_output, left, right);

//     auto end = std::chrono::steady_clock::now();
//     int fwd_time = static_cast<int>(std::chrono::duration<double,
//                                     std::micro>(end - start).count());

//     std::cout << " " << fwd_time << " ms" << std::endl;


//     // Copy output from device to host
//     cudaMemcpy(output, d_output, right * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_input);
//     cudaFree(d_weights);
//     cudaFree(d_output);
// }
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

    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, left * sizeof(float));
    cudaMalloc(&d_B, left * right * sizeof(float));
    cudaMalloc(&d_C, right * sizeof(float));

    cudaMemcpy(d_A, input, left * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, left * right * sizeof(float), cudaMemcpyHostToDevice);



    auto start = std::chrono::steady_clock::now();

    // Do the actual multiplication
    sgemm_naive<<<64, 64>>>(m, n, k, 1.0, d_A, d_B, 0.0, d_C);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                    std::micro>(end - start).count());

    std::cout << " " << fwd_time << " ms" << std::endl;

    cudaMemcpy(output, d_C, right * sizeof(float), cudaMemcpyDeviceToHost);
    // Destroy the handle
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess) 
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }



// void cuFC(float *input, float *output, int left, int right)
// {
//     float *d_input, *d_weights, *d_output;
//     gpuErrchk(cudaMalloc(&d_input, left * sizeof(float)));
//     gpuErrchk(cudaMalloc(&d_weights, left * right * sizeof(float)));
//     gpuErrchk(cudaMalloc(&d_output, right * sizeof(float)));

//     gpuErrchk(cudaMemcpy(d_input, input, left * sizeof(float), cudaMemcpyHostToDevice));

//     float *h_weights = (float *)malloc(left * right * sizeof(float));
//     for (int i = 0; i < left * right; i++) {
//         h_weights[i] = (float)std::rand() / RAND_MAX / 1000;
//     }
//     gpuErrchk(cudaMemcpy(d_weights, h_weights, left * right * sizeof(float), cudaMemcpyHostToDevice));
//     free(h_weights);

//     dim3 threadsPerBlock(256);
//     dim3 numBlocks((right + threadsPerBlock.x - 1) / threadsPerBlock.x);

//     auto start = std::chrono::steady_clock::now();
//     matrixVectorMultiply<<<numBlocks, threadsPerBlock>>>(d_weights, d_input, d_output, left, right);
//     gpuErrchk(cudaPeekAtLastError());
//     gpuErrchk(cudaDeviceSynchronize()); // Ensure the kernel completes and captures any errors

//     auto end = std::chrono::steady_clock::now();
//     std::chrono::duration<double, std::micro> elapsed = end - start;
//     std::cout << "Kernel Execution Time: " << elapsed.count() << " us" << std::endl;

//     gpuErrchk(cudaMemcpy(output, d_output, right * sizeof(float), cudaMemcpyDeviceToHost));

//     cudaFree(d_input);
//     cudaFree(d_weights);
//     cudaFree(d_output);
// }



int main()
{
    std::srand(std::time(0));

    float *input;
    float *output;

    int data_size = 224 * 224 * 3 * 1;
    input = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; i++)
    {
        input[i] = (float)std::rand() / RAND_MAX;
    }


    
    // Block 1
    std::cout << "CONV 224x224x64";
    output = (float *)malloc(224 * 224 * 64 * 1 * sizeof(float));
    cuConv2D(input, output, 224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 224 * 224 * 64);

    std::swap(input, output);
    free(output);

    std::cout << "CONV 224x224x64";
    output = (float *)malloc(224 * 224 * 64 * 1 * sizeof(float));
    cuConv2D(input, output, 224, 224, 64, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 224 * 224 * 64);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 112x112x64";
    output = (float *)malloc(112 * 112 * 64 * sizeof(float));
    cuMaxPool(input, output, 224, 224, 64, 1);
    std::swap(input, output);
    free(output);

    // Block 2
    std::cout << "CONV 112x112x128";
    output = (float *)malloc(112 * 112 * 128 * 1 * sizeof(float));
    cuConv2D(input, output, 112, 112, 64, 1, 128, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 112 * 112 * 128);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 112x112x128";
    output = (float *)malloc(112 * 112 * 128 * 1 * sizeof(float));
    cuConv2D(input, output, 112, 112, 128, 1, 128, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 112 * 112 * 128);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 56x56x128";
    output = (float *)malloc(56 * 56 * 128 * sizeof(float));
    cuMaxPool(input, output, 112, 112, 128, 1);
    std::swap(input, output);
    free(output);

    // Block 3
    std::cout << "CONV 56x56x256";
    output = (float *)malloc(56 * 56 * 256 * 1 * sizeof(float));
    cuConv2D(input, output, 56, 56, 128, 1, 256, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 56 * 56 * 256);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 56x56x256";
    output = (float *)malloc(56 * 56 * 256 * 1 * sizeof(float));
    cuConv2D(input, output, 56, 56, 256, 1, 256, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 56 * 56 * 256);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 56x56x256";
    output = (float *)malloc(56 * 56 * 256 * 1 * sizeof(float));
    cuConv2D(input, output, 56, 56, 256, 1, 256, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 56 * 56 * 256);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 28x28x256";
    output = (float *)malloc(28 * 28 * 256 * sizeof(float));
    cuMaxPool(input, output, 56, 56, 256, 1);
    std::swap(input, output);
    free(output);

    // Block 4
    std::cout << "CONV 28x28x512";
    output = (float *)malloc(28 * 28 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 28, 28, 256, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 28 * 28 * 512);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 28x28x512";
    output = (float *)malloc(28 * 28 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 28, 28, 512, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 28 * 28 * 512);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 28x28x512";
    output = (float *)malloc(28 * 28 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 28, 28, 512, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 28 * 28 * 512);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 14x14x512";
    output = (float *)malloc(14 * 14 * 512 * sizeof(float));
    cuMaxPool(input, output, 28, 28, 512, 1);
    std::swap(input, output);
    free(output);

    // Block 5
    std::cout << "CONV 14x14x512";
    output = (float *)malloc(14 * 14 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 14 * 14 * 512);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 14x14x512";
    output = (float *)malloc(14 * 14 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 14 * 14 * 512);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 14x14x512";
    output = (float *)malloc(14 * 14 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1, 1, 1);
    cuReLU(output, output, 14 * 14 * 512);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 7x7x512";
    output = (float *)malloc(7 * 7 * 512 * sizeof(float));
    cuMaxPool(input, output, 14, 14, 512, 1);
    std::swap(input, output);
    free(output);

    // Fully connected layers
    std::cout << "FC 4096";
    output = (float *)malloc(4096 * sizeof(float));
    cuFC(input, output, 7 * 7 * 512, 4096);
    cuReLU(output, output, 4096);
    std::swap(input, output);
    free(output);

    std::cout << "FC 4096";
    output = (float *)malloc(4096 * sizeof(float));
    cuFC(input, output, 4096, 4096);
    cuReLU(output, output, 4096);
    std::swap(input, output);
    free(output);

    std::cout << "FC 1000";
    output = (float *)malloc(1000 * sizeof(float));
    cuFC(input, output, 4096, 1000);

    free(input);
    free(output);

    return 0;
}