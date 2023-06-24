#include <stdio.h>
#include "cuda_utils.h"
#include <iostream>
#include <chrono>

__global__
void rrrt_kernel(in_t* input, out_t *output, 
                int     in_x , int   in_y   , 
                int     out_x, int   out_y  , 
                float   mxx  , float myx    , 
                float   mxy  , float myy    , 
                float   tx   , float ty     ){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
  
    int stride = blockDim.x * gridDim.x;

    int out_idx = index;
    int in_idx;

    int i_out = 0, j_out = 0;

    float i_in = 0, j_in = 0;

    float i_in_m = 0, j_in_m = 0;

    int counter = 0;

    while (out_idx < out_y*out_x){

        counter ++;

        j_out = int(out_idx/out_x);
        i_out = int(out_idx - j_out*out_x);

        i_in = i_out;
        j_in = j_out;

        i_in -= out_x/2.0;
        j_in -= out_y/2.0;

        i_in_m = i_in*mxx + j_in*mxy;
        j_in_m = i_in*myx + j_in*myy;

        i_in_m += in_x/2.0 + tx;
        j_in_m += in_y/2.0 + ty;

        in_idx = round(i_in_m) + in_x*round(j_in_m);

        if (i_in_m < in_x && j_in_m < in_y && i_in_m >= 0 && j_in_m >= 0){
            output[out_idx] = (out_t)input[in_idx];
        }

        out_idx += stride;
    }

}


void alloc_in_out_cuda(in_t*  input, in_t** in_cuda, out_t*** out_cuda, int out_n, int N_in, int N_out){
    auto error = cudaMallocManaged(in_cuda, N_in*sizeof(uchar));

    std::cout << cudaGetErrorString(error) << '\n';

    cudaMemcpy(*in_cuda, input, N_in*sizeof(in_t), cudaMemcpyHostToDevice); 

    cudaMallocManaged(out_cuda, N_out*sizeof(out_t*));
        
    for (int i = 0; i < out_n; i++){
        cudaMallocManaged(&((*out_cuda)[i]), N_out*sizeof(out_t));
    }
}


void free_in_out_cuda(in_t* in_cuda, out_t** out_cuda, int out_n){
    
    cudaFree(in_cuda);

    for (int i = 0; i < out_n; i++){
        cudaFree(out_cuda[i]);
    }

    cudaFree(out_cuda);
}

void cuda_out_to_host(out_t** out_cuda, out_t** out_host, int out_n, int out_x, int out_y){

    for (int i = 0; i < out_n; i++){
        out_host[i] = (out_t*)calloc(out_x*out_y, sizeof(out_t));
        cudaMemcpy(out_host[i], out_cuda[i], out_x*out_y*sizeof(out_t), cudaMemcpyDeviceToHost);
    }

}

void rrrt_cuda(in_t*  in_cuda, out_t** out_cuda, 
               int     out_n,
               int     in_x , int     in_y  , 
               int     out_x, int     out_y , 
               float*  mxx  , float*  myx   , 
               float*  mxy  , float*  myy   , 
               float*  tx   , float*  ty    ){

    
    int N_out = out_x*out_y;

    int blockSize = 256;
    int numBlocks = (N_out + blockSize - 1) / blockSize;

    for (int i = 0; i < out_n; i++){
        rrrt_kernel<<<numBlocks, blockSize>>>(in_cuda, out_cuda[i], in_x, in_y, out_x, out_y, mxx[i], myx[i], mxy[i],  myy[i], tx[i], ty[i]);
    }

    cudaDeviceSynchronize();

}

void rrrt_host(in_t*  in_host, out_t** out_host, 
          int     out_n,
          int     in_x , int     in_y  , 
          int     out_x, int     out_y , 
          float*  mxx  , float*  myx   , 
          float*  mxy  , float*  myy   , 
          float*  tx   , float*  ty    ){
        
    int N_in  = in_x*in_y;
    int N_out = out_x*out_y;

    in_t    *in_cuda;
    out_t **out_cuda;
    
    alloc_in_out_cuda(in_host, &in_cuda, &out_cuda, out_n, N_in, N_out);

    rrrt_cuda(in_cuda, out_cuda, out_n, in_x , in_y , out_x, out_y , mxx  , myx , mxy  , myy   ,tx   , ty );

    cuda_out_to_host(out_cuda, out_host, out_n, out_x, out_y);

    free_in_out_cuda(in_cuda, out_cuda, out_n);

}
