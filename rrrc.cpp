#include <torch/extension.h>
//#include <stdio.h>
#include <iostream>


typedef unsigned char in_t;
typedef float out_t; 


#define CHECK_CUDA(x)       TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void rrrt_cuda(in_t*  in_cuda, out_t* out_cuda, 
               int     out_n,
               int     in_x , int     in_y  , 
               int     out_x, int     out_y , 
               float*  M,     float*  T);

torch::Tensor rrrc_torch(torch::Tensor in_image, torch::Tensor out_images, torch::Tensor M, torch::Tensor T) {
  
  std::cout<< "here" << std::endl;


  CHECK_INPUT(in_image);
  CHECK_INPUT(out_images);

  auto m = (float*)M.data_ptr<float>();
  auto t = (float*)T.data_ptr<float>();

  //printf("HERE");
  int in_x  = in_image.sizes()[1];
  int in_y  = in_image.sizes()[2];

  int out_n  = out_images.sizes()[0];
  int out_x  = out_images.sizes()[2];
  int out_y  = out_images.sizes()[3];

  std::cout << in_x << " " << in_y << " " << out_x << " " << out_y << " " << out_n << std::endl;

  auto in_cuda  = (in_t*)in_image.flatten().contiguous().data_ptr<in_t>();
  auto out_cuda = (out_t*)out_images.flatten().contiguous().data_ptr<out_t>();

  rrrt_cuda(in_cuda, out_cuda, out_n, in_x , in_y  , out_x, out_y , m, t);


  /*for (int i = 0; i <= 10; i++){
    out_cuda[i] = 0.0;
  }*/


  return out_images;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rrrc", &rrrc_torch, "LLTM forward");
}


