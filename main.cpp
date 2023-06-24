// c++ DeepAirNav

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>
#include <chrono>
#include <cmath>


#include "cuda_utils.h"

//nvcc -c cuda_utils.cu -O3
//g++ -o main main.cpp -O3 -I/usr/include/opencv4/ cuda_utils.o -lcuda -lcudart -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc 

int main(){

    cv::Mat image = cv::imread( "./0-0.png", cv::IMREAD_COLOR );
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    image.convertTo(image, CV_8UC1); //------------
    image = image(cv::Range(0,  1972), cv::Range(0,  1972));
    image = image.clone();

    std::cout << "Width:    " << image.cols       << std::endl;
    std::cout << "Height:   " << image.rows       << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;

    int out_x = 140, out_y = 140;
    int in_x = image.cols, in_y = image.rows;
    int out_n = 1000;

    float mxx[1000], mxy[1000], myx[1000], myy[1000], tx[1000], ty[1000];

    srand(time(0));

    for (int i = 0; i< out_n; i++){
        int alpha = rand();
        int a = (rand()%100)/10.0;
        mxx[i] =  cos(alpha)/a  * in_x/out_x;
        mxy[i] =  sin(alpha)/a  * in_x/out_x;
        myx[i] = -sin(alpha)/a * in_y/out_y;
        myy[i] =  cos(alpha)/a * in_y/out_y;
        tx[i]  =  rand()%600 - 300;
        ty[i]  =  rand()%600 - 300;
    }

    out_t** output = (out_t**) calloc(out_n, sizeof(out_t*)); //------------

    for (int i = 0; i < out_n; i++){
        output[i] = (out_t*) calloc(out_x*out_y, sizeof(out_t)); //------------
    }

    int N_in  = in_x *in_y;
    int N_out = out_x*out_y;

    in_t* in_host    =  image.data;
    out_t** out_host =  output;

    in_t *in_cuda = 0;
    out_t **out_cuda = 0;


    auto start = std::chrono::steady_clock::now();
    alloc_in_out_cuda(in_host, &in_cuda, &out_cuda, out_n, N_in, N_out);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "alloc_in_out_cuda time: " << elapsed_seconds.count() << "s\n";


    start = std::chrono::steady_clock::now();
    rrrt_cuda(in_cuda, out_cuda, out_n, in_x , in_y , out_x, out_y , mxx  , myx , mxy  , myy   ,tx   , ty );
    end = std::chrono::steady_clock::now();

    elapsed_seconds = end-start;
    std::cout << "rrrt_cuda time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::steady_clock::now();
    cuda_out_to_host(out_cuda, out_host, out_n, out_x, out_y);
    end = std::chrono::steady_clock::now();

    elapsed_seconds = end-start;
    std::cout << "cuda_out_to_host time: " << elapsed_seconds.count() << "s\n";


    cv::namedWindow("in", cv::WINDOW_AUTOSIZE );
    cv::imshow("in", image);
    std::string str = "out";

    cv::Mat out_image(out_x,out_y,CV_8UC1);

    for (int i = 0; i < 1000; i+=200){

        in_t* output_in_t = (in_t*)calloc(N_out, sizeof(in_t));
        for (int j = 0; j < N_out; j++){
            output_in_t[j] = (in_t)output[i][j];
        }

        out_image.data = (output_in_t);
        str.append("*");
        cv::namedWindow(str, cv::WINDOW_AUTOSIZE );
        cv::imshow(str, out_image);
    }

    cv::waitKey(0);

    return 0;
}



