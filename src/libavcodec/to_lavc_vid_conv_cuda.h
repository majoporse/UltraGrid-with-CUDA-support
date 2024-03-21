#ifndef LIBAVCODEC_TO_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9
#define LIBAVCODEC_TO_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9

#include "../config_unix.h"
#include "../video_codec.h"

#include "../libavcodec/lavc_common.h"
#include "../video_codec.h"
#include <libavutil/pixdesc.h>

#include "libavutil/pixfmt.h"
#include "../config_unix.h"
#include "../video_codec.h"

#include "../libavcodec/lavc_common.h"
#include "../video_codec.h"
#include "cuda_utils.h"
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
//#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

static const enum AVPixelFormat to_lavc_cuda_supp_formats[] = {
        AV_PIX_FMT_YUV420P10LE,
        AV_PIX_FMT_YUV444P10LE,
        AV_PIX_FMT_YUV422P10LE,
        AV_PIX_FMT_P010LE,
        AV_PIX_FMT_NV12,
        AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUVJ422P,
        AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_YUV420P12LE,
        AV_PIX_FMT_YUV422P12LE,
        AV_PIX_FMT_YUV444P12LE,
        AV_PIX_FMT_YUV420P16LE,
        AV_PIX_FMT_YUV422P16LE,
        AV_PIX_FMT_YUV444P16LE,
        AV_PIX_FMT_AYUV64,
        AV_PIX_FMT_GBRP,
        AV_PIX_FMT_GBRAP,
        AV_PIX_FMT_GBRP10LE,
        AV_PIX_FMT_GBRP12LE,
        AV_PIX_FMT_GBRP16LE,
        AV_PIX_FMT_GBRAP10LE,
        AV_PIX_FMT_GBRAP12LE,
        AV_PIX_FMT_GBRAP16LE,
        AV_PIX_FMT_BGR0,
        AV_PIX_FMT_BGRA,
        AV_PIX_FMT_RGB24,
        AV_PIX_FMT_RGB48LE,
        AV_PIX_FMT_RGBA64LE,
        AV_PIX_FMT_RGBA,
        AV_PIX_FMT_Y210,
#if P210_PRESENT
        AV_PIX_FMT_P210LE,
#endif
#if XV3X_PRESENT
        AV_PIX_FMT_XV30,
        AV_PIX_FMT_Y212,
#endif
#if VUYX_PRESENT
        AV_PIX_FMT_VUYA,
        AV_PIX_FMT_VUYX,
#endif
#if X2RGB10LE_PRESENT
        AV_PIX_FMT_X2RGB10LE,
#endif
        AV_PIX_FMT_NONE
};



typedef struct {
    AVFrame *frame;
    codec_t to;
    char *intermediate_to;
    char *gpu_in_buffer;
    AVFrame *gpu_frame;
    AVF_GPU_wrapper *gpu_wrapper;
} to_lavc_conv_cuda;

// #define HAVE_CUDA
#ifdef HAVE_LAVC_CUDA_CONV

AVFrame *to_lavc_vid_conv_cuda(to_lavc_conv_cuda *, const char *src);

to_lavc_conv_cuda *to_lavc_vid_conv_cuda_init(enum AVPixelFormat, codec_t, int, int);

void to_lavc_vid_conv_cuda_destroy(to_lavc_conv_cuda **);

#else
#include <stdio.h>

to_lavc_conv_cuda *to_lavc_vid_conv_cuda_init(enum AVPixelFormat f, codec_t c, int w, int h)
{
    (void) f, (void) c, (void) w, (void) h;
    fprintf(stderr, "ERROR: CUDA support not compiled in!\n");
    return NULL;
}

AVFrame *to_lavc_vid_conv_cuda(to_lavc_conv_cuda *s, const char* p)
{
    (void) s, (void) p;
    return NULL;
}

void to_lavc_vid_conv_cuda_destroy(to_lavc_conv_cuda **state)
{
    (void) state;
}
#endif

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_TO_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9
