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

#ifdef __cplusplus
extern "C" {
#endif

/// @note needs to support conversion for all dst codec_t
static const enum AVPixelFormat from_lavc_cuda_supp_formats[] = {
        AV_PIX_FMT_YUV422P
};

struct av_to_uv_convert_cuda;

#ifdef HAVE_CUDA
struct av_to_uv_convert_cuda *
get_av_to_uv_cuda_conversion(enum AVPixelFormat av_codec, codec_t uv_codec);
void av_to_uv_convert_cuda(struct av_to_uv_convert_cuda *state,
                           char *__restrict dst_buffer,
                           struct AVFrame *__restrict in_frame, int width,
                           int height, int pitch,
                           const int *__restrict rgb_shift);
void av_to_uv_conversion_cuda_destroy(struct av_to_uv_convert_cuda **state);

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
