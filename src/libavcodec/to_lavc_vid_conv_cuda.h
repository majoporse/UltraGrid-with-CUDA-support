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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
//#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

<<<<<<< HEAD
/// @note needs to support conversion for all src codec_t
static const enum AVPixelFormat to_lavc_cuda_supp_formats[] = { AV_PIX_FMT_YUV444P };

#ifdef HAVE_LAVC_CUDA_CONV
struct to_lavc_vid_conv_cuda *
to_lavc_vid_conv_cuda_init(codec_t in_pixfmt, int width, int height,
                           enum AVPixelFormat out_pixfmt);
struct AVFrame *to_lavc_vid_conv_cuda(struct to_lavc_vid_conv_cuda *state,
                                      const char                   *in_data);
void to_lavc_vid_conv_cuda_destroy(struct to_lavc_vid_conv_cuda **state);
=======
#define HAVE_CUDA
#ifdef HAVE_CUDA
#include "cuda_utils.h"

typedef struct {
    AVFrame *frame;
    codec_t to;
    char *intermediate_to;
    char *gpu_in_buffer;
    AVFrame *gpu_frame;
    AVF_GPU_wrapper *gpu_wrapper;
} to_lavc_conv_cuda;

AVFrame *to_lavc_vid_conv_cuda(to_lavc_conv_cuda *, const char *src);

to_lavc_conv_cuda *to_lavc_vid_conv_cuda_init(enum AVPixelFormat, codec_t, int, int);

void to_lavc_vid_conv_cuda_destroy(to_lavc_conv_cuda **);
>>>>>>> abc5c294b (removed c++20 features + some interface changes to work with UG)

#else
#include <stdio.h>

struct to_lavc_conv_cuda{};

struct to_lavc_conv_cuda *to_lavc_vid_conv_cuda_init(enum AVPixelFormat f, codec_t c, int w, int h)
{
    (void) f, (void) c, (void) w, (void) h;
    fprintf(stderr, "ERROR: CUDA support not compiled in!\n");
    return NULL;
}

AVFrame *to_lavc_vid_conv_cuda(struct to_lavc_conv_cuda s, const char* p)
{
    (void) s, (void) p;
    return NULL;
}

void to_lavc_vid_conv_cuda_destroy(struct to_lavc_vid_conv_cuda **state)
{
    (void) state;
}
#endif

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_TO_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9
