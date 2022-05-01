
#include <chrono>
#include <iostream>
#include <CL/sycl.hpp>

#define random_float() (rand() / double(RAND_MAX)) 

using namespace std;
using namespace sycl;



// return execution time
double gpu_kernel(float* X, int* mask, float* weight, float* loss, int K, int M, int N, int block_size, sycl::queue& q) {

    //define the workgroup size and mapping
    auto grid_rows = (M + block_size - 1) / block_size * block_size;
    auto grid_cols = (N + block_size - 1) / block_size * block_size;
    auto grid_height = (K + block_size - 1) / block_size * block_size;
    auto local_ndrange = range<2>(block_size, block_size);
    auto global_ndrange = range<2>(grid_height, grid_cols);

    auto xmaxi = malloc_shared<float>(K * N, q);
    auto sum = malloc_shared<float>(K * N, q);
    auto y = malloc_shared<float>(K * N, q);
    for (int i = 0; i < K * N; i++) {
        xmaxi[i] = sum[i] = y[i] = 0.0f;
    }

    double duration = 0.0f;
    // get the x_max parallelly
    auto e = q.submit([ & ](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global_ndrange, local_ndrange), [ = ](sycl::nd_item<2> index) {
                int layer = index.get_local_id(0) + index.get_group(0) * block_size;
                int col = index.get_local_id(1) + index.get_group(1) * block_size;
                for (int i = 0; i < M; i++) {
                    xmaxi[layer * N + col] =
                        std::max(xmaxi[layer * N + col], X[layer * M * N + col + i * N]);
                }
            });
        });
    e.wait();
    //get the sum parallelly
    e = q.submit([ & ](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global_ndrange, local_ndrange), [ = ](sycl::nd_item<2> index) {
                int layer = index.get_local_id(0) + index.get_group(0) * block_size;
                int col = index.get_local_id(1) + index.get_group(1) * block_size;
                for (int i = 0; i < M; i++) {
                    sum[layer * N + col] += exp(X[layer * N * M + col + i * N] - xmaxi[layer * N + col]);
                }
            });
        });
    e.wait();

    e = q.submit([ & ](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global_ndrange, local_ndrange), [ = ](sycl::nd_item<2> index) {
                int layer = index.get_local_id(0) + index.get_group(0) * block_size;
                int col = index.get_local_id(1) + index.get_group(1) * block_size;
                // only calculate y for the choosen element 
                y[layer * N + col] = X[layer * M * N + col + N * mask[layer * N + col]]
                    - xmaxi[layer * N + col] - log(sum[layer * N + col]);
                loss[layer * N + col] = -y[layer * N + col] * weight[layer * N + col];
            });
        });
    e.wait();

    duration += (e.get_profiling_info<info::event_profiling::command_end>() -
        e.get_profiling_info<info::event_profiling::command_start>()) / 1000.0f / 1000.0f;

    free(xmaxi, q);
    free(sum, q);
    free(y, q);
    return(duration);
}

// return execution time
double cpu_kernel(float* cX, int* cmask, float* weight, float* loss_host, int K, int M, int N) {

    double duration = 0.0;
    std::chrono::high_resolution_clock::time_point s, e;

    float* y = new float[K * M * N];
    // Single Thread Computation in CPU 
    s = std::chrono::high_resolution_clock::now();


    for (int h = 0; h < K; h++) {
        for (int i = 0; i < N; i++) {
            // find the xmax of each column
            auto xmaxi = -1.0f;
            auto sum = 0.f;
            for (int j = 0; j < M; j++)
                xmaxi = std::max(xmaxi, cX[h * M * N + i + j * N]);
            for (int j = 0; j < M; j++)
                sum += exp(cX[h * M * N + i + j * N] - xmaxi);
            // calculate y for all the elements
            for (int j = 0; j < M; j++) {
                y[h * M * N + i + j * N] =
                    cX[h * M * N + i + j * N] - xmaxi - log(sum);
            }
            loss_host[h * N + i] = -y[h * M * N + i + N * cmask[h * N + i]] * weight[h * N + i];
        }
    }

    e = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(e - s).count();
    delete[] y;
    return(duration);
}

int verify(float* cpu_res, float* gpu_res, int length) {
    int err = 0;
    for (int i = 0; i < length; i++) {
        if (fabs(cpu_res[i] - gpu_res[i]) > 1e-3) {
            err++;
            printf("\n%lf, %lf", cpu_res[i], gpu_res[i]);
        }
    }
    return(err);
}

int Cross_Entropy(const int K,
    const int M,
    const int N,
    const int block_size,
    const int iterations,
    sycl::queue& q) {

    cout << "Problem size " << K << " " << M << " " << N;

    auto X = malloc_shared<float>(K * M * N, q);
    auto mask = malloc_shared<int>(K * N, q);
    auto weight = malloc_shared<float>(K * N, q);
    auto loss = malloc_shared<float>(K * N, q);
    auto loss_host = malloc_host<float>(K * N, q);

    // init the X, mask, weight,loss, loss_host
    for (int i = 0; i < K * M * N; i++) {
        X[i] = random_float();
    }

    for (int i = 0; i < K * N; i++) {
        mask[i] = (int)rand() % M;
        weight[i] = random_float();
    }

    for (int i = 0; i < M * N; i++) {
        loss[i] = 0.0f;
        loss_host[i] = 0.0f;
    }

    double flopsPerMatrixMul
        = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    double duration_gpu = 0.0f;
    double duration_cpu = 0.0f;

    // GPU compuation and timer 
    int warmup = 10;
    for (int run = 0; run < iterations + warmup; run++) {
        float duration = gpu_kernel(X, mask, weight, loss, K, M, N, block_size, q);
        if (run >= warmup) duration_gpu += duration;
    }
    duration_gpu = duration_gpu / iterations;

    // CPU compuation and timer 
    warmup = 2;
    for (int run = 0; run < iterations / 2 + warmup; run++) {
        float duration = cpu_kernel(X, mask, weight, loss_host, K, M, N);
        if (run >= warmup) duration_cpu += duration;
    }
    duration_cpu = duration_cpu / iterations / 2;

    // Compare the resutls of CPU and GPU 
    int errCode = 0;
    errCode = verify(loss_host, loss, K * N);
    if (errCode > 0) printf("\nThere are %d errors\n", errCode);

    printf("\nPerformance Flops = %lf, \n"
        "GPU Computation Time = %lf (ms); \n"
        "CPU Computaiton Time = %lf (ms); \n",
        flopsPerMatrixMul, duration_gpu, duration_cpu);

    //get some view

    //std::cout << std::endl;
    //for (int i = 0; i <  N; i++)
    //    std::cout << mask[N + i] << " ";
    //std::cout << std::endl;
    //for (int i = 0; i < M * N; i++) {
    //    if (i % N==0) std::cout << std::endl;
    //    std::cout << X[i+N*M] << " ";
    //}
    //std::cout << std::endl << std::endl;
    //for (int i = 0; i <  N; i++) {
    //    //if (i % N == 0) std::cout << std::endl;
    //    std::cout << weight[i+N] << " ";
    //}
    //std::cout << std::endl;

    free(X, q);
    free(mask, q);
    free(weight, q);
    free(loss, q);
    free(loss_host, q);
    return(errCode);
}

int main() {

    auto propList = cl::sycl::property_list{ cl::sycl::property::queue::enable_profiling() };
    queue my_gpu_queue(cl::sycl::gpu_selector{}, propList);

    int errCode = Cross_Entropy(128, 32, 8192, 7, 10, my_gpu_queue);

    return(errCode);
}
