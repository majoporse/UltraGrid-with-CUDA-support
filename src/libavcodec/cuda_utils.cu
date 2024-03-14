#include "cuda_utils.h"
#include <iostream>

extern "C" void alloc(AVF_GPU_wrapper *wrapper, const AVFrame* new_frame) {
    AVFrame *frame = &(wrapper->frame);
    *frame = *new_frame;
    wrapper->q = 1;
    if (new_frame->format == AV_PIX_FMT_YUV420P ||
        new_frame->format == AV_PIX_FMT_YUV420P10LE ||
        new_frame->format == AV_PIX_FMT_YUV420P12LE ||
        new_frame->format == AV_PIX_FMT_YUV420P16LE ||
        new_frame->format == AV_PIX_FMT_NV12 ||
        new_frame->format == AV_PIX_FMT_YUVJ420P ||
        new_frame->format == AV_PIX_FMT_P010LE)
        wrapper->q = 2;
    cudaMalloc(&(frame->data[0]), frame->linesize[0] * frame->height);
    cudaMalloc(&(frame->data[1]), frame->linesize[1] * frame->height / wrapper->q);
    cudaMalloc(&(frame->data[2]), frame->linesize[2] * frame->height / wrapper->q);
    cudaMalloc(&(frame->data[3]), frame->linesize[3] * frame->height);
}

extern "C" void copy_to_device(AVF_GPU_wrapper *wrapper, const AVFrame *new_frame){
    AVFrame *frame = &(wrapper->frame);
                            cudaMemcpy(frame->data[0], new_frame->data[0], frame->linesize[0] * frame->height, cudaMemcpyHostToDevice);
    if (frame->linesize[1]) cudaMemcpy(frame->data[1], new_frame->data[1], frame->linesize[1] * frame->height / wrapper->q, cudaMemcpyHostToDevice);
    if (frame->linesize[2]) cudaMemcpy(frame->data[2], new_frame->data[2], frame->linesize[2] * frame->height / wrapper->q, cudaMemcpyHostToDevice);
    if (frame->linesize[3]) cudaMemcpy(frame->data[3], new_frame->data[3], frame->linesize[3] * frame->height, cudaMemcpyHostToDevice);
}

extern "C" void copy_to_host(AVF_GPU_wrapper *wrapper, const AVFrame *new_frame){
    AVFrame *frame = &(wrapper->frame);
                            cudaMemcpy( new_frame->data[0],frame->data[0], frame->linesize[0] * frame->height, cudaMemcpyDeviceToHost);
    if (frame->linesize[1]) cudaMemcpy( new_frame->data[1],frame->data[1], frame->linesize[1] * frame->height / wrapper->q, cudaMemcpyDeviceToHost);
    if (frame->linesize[2]) cudaMemcpy( new_frame->data[2],frame->data[2], frame->linesize[2] * frame->height / wrapper->q, cudaMemcpyDeviceToHost);
    if (frame->linesize[3]) cudaMemcpy( new_frame->data[3],frame->data[3], frame->linesize[3] * frame->height, cudaMemcpyDeviceToHost);
}

extern "C" void free_from_device(AVF_GPU_wrapper *wrapper){
    AVFrame *frame = &(wrapper->frame);
    cudaFree(frame->data[0]);
    cudaFree(frame->data[1]);
    cudaFree(frame->data[2]);
    cudaFree(frame->data[3]);

    frame->data[0] = nullptr;
    frame->data[1] = nullptr;
    frame->data[2] = nullptr;
    frame->data[3] = nullptr;
    wrapper->q = 1;
}