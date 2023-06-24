typedef unsigned char uchar;

typedef unsigned char in_t;
typedef float out_t;


void rrrt_host(in_t*  input, out_t** output,
          int     out_n,
          int     in_x , int     in_y  , 
          int     out_x, int     out_y , 
          float*  mxx  , float*  myx   , 
          float*  mxy  , float*  myy   , 
          float*  tx   , float*  ty    );

void rrrt_cuda(in_t*  in_cuda, out_t** out_cuda, 
               int     out_n,
               int     in_x , int     in_y  , 
               int     out_x, int     out_y , 
               float*  mxx  , float*  myx   , 
               float*  mxy  , float*  myy   , 
               float*  tx   , float*  ty    );

void cuda_out_to_host(out_t** out_cuda, out_t** out_host, int out_n, int out_x, int out_y);

void free_in_out_cuda(in_t* in_cuda, out_t** out_cuda, int out_n);

void alloc_in_out_cuda(in_t*  input, in_t** in_cuda, out_t*** out_cuda, int out_n, int N_in, int N_out);

