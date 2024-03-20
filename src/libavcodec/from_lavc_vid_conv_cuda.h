#ifndef LIBAVCODEC_FROM_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9
#define LIBAVCODEC_FROM_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "libavutil/pixfmt.h"
#include "../config_unix.h"
#include "../video_codec.h"

#include "../libavcodec/lavc_common.h"
#include "../video_codec.h"
#include "cuda_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

static const enum AVPixelFormat from_lavc_cuda_supp_formats[] = {
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
        AV_PIX_FMT_NONE
};

typedef struct {
    char * ptr;
    codec_t to;
    char *intermediate;
    char *gpu_out_buffer;
    AVF_GPU_wrapper *wrapper;
    AVFrame *gpu_frame;
} from_lavc_conv_state;

// #define HAVE_CUDA
#ifdef HAVE_CUDA

char *av_to_uv_convert_cuda(from_lavc_conv_state *state, const AVFrame* frame);

 from_lavc_conv_state *av_to_uv_conversion_cuda_init(const AVFrame*, codec_t);

void av_to_uv_conversion_cuda_destroy(from_lavc_conv_state **);
#else

char * av_to_uv_convert_cuda(from_lavc_conv_state *state,  const AVFrame* frame){
    (void) state; (void) frame;
    return NULL;
}

from_lavc_conv_state *av_to_uv_conversion_cuda_init(const AVFrame *f, codec_t c){
    (void) f; (void) c;
    return NULL;
}

void av_to_uv_conversion_cuda_destroy(from_lavc_conv_state **s){
    (void) s;
}
#endif

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_FROM_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9
