
typedef unsigned char in_t;
typedef float out_t; 

#include <iostream>
#include <chrono>




__global__
__forceinline__
void rrrt_kernel(in_t* __restrict__ input, out_t* __restrict__ output, 
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



void rrrt_cuda(in_t* __restrict__ in_cuda, out_t* __restrict__ out_cuda, 
               int     out_n,
               int     in_x , int     in_y  , 
               int     out_x, int     out_y , 
               float* __restrict__ M,     float* __restrict__ T){


    int N_out = out_x*out_y;

    int blockSize = 256;
    int numBlocks = (N_out + blockSize - 1) / blockSize;
    
    

    for (int i = 0; i < out_n; i++){
        
        rrrt_kernel<<<numBlocks, blockSize>>>(in_cuda, out_cuda + i*N_out, in_x, in_y, out_x, out_y, M[4*i], M[4*i + 1], M[4*i + 2],  M[4*i + 3], T[2*i], T[2*i + 1]);

    }

    /*auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";*/

    cudaDeviceSynchronize();
}