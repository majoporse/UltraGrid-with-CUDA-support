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

typedef struct {
    char * ptr;
    codec_t to;
    char *intermediate;
    char *gpu_out_buffer;
    AVF_GPU_wrapper *wrapper;
    AVFrame *gpu_frame;
} from_lavc_conv_state;

#define HAVE_CUDA
#ifdef HAVE_CUDA

char *av_to_uv_convert_cuda(from_lavc_conv_state *state, const AVFrame* frame);

 from_lavc_conv_state *av_to_uv_conversion_cuda_init(const AVFrame*, codec_t);

void av_to_uv_conversion_cuda_destroy(from_lavc_conv_state **);
#else

typedef struct{}from_lavc_conv_state;

char * convert_from_lavc(from_lavc_conv_state *state,  const AVFrame* frame){
    (void) state; (void) frame;
    return NULL;
}

from_lavc_conv_state *from_lavc_init(const AVFrame *f, codec_t c){
    (void) f; (void) c;
    return NULL;
}

void from_lavc_destroy(from_lavc_conv_state **s){
    (void) s;
}
#endif

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_FROM_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9
