#ifndef CONV_UTILS
#define CONV_UTILS

#include <libavutil/pixfmt.h>
#include <libavutil/frame.h>

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

typedef struct {
    AVFrame frame;
    int q;
} AVF_GPU_wrapper;

void alloc(AVF_GPU_wrapper* wrapper, const AVFrame* new_frame);

void copy_to_device(AVF_GPU_wrapper* wrapper, const AVFrame *new_frame);

void copy_to_host(AVF_GPU_wrapper* wrapper, const AVFrame *new_frame);

void free_from_device(AVF_GPU_wrapper* wrapper);

#endif
