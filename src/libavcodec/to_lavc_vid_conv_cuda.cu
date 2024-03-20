#include "to_lavc_vid_conv_cuda.h"
#include <map>
#include <algorithm>
#include <iostream>
#include "../color.h"
#include <array>
#include "cuda_utils.h"

#define YUV_INTER_TO 0
#define RGB_INTER_TO 1

#define BLOCK_SIZE 32

/**************************************************************************************************************/
/*                                            KERNELS FROM                                                    */
/**************************************************************************************************************/

template< typename OUT_T, int bit_shift, bool has_alpha>
__global__ void convert_rgbp_from_inter(int width, int height, size_t pitch_in, char *in, AVFrame *frame){
    //RGBA 16bit -> rgb AVF
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_g_row = frame->data[0] + frame->linesize[0] *  y;
    void *dst_b_row = frame->data[1] + frame->linesize[1] *  y;
    void *dst_r_row = frame->data[2] + frame->linesize[2] *  y;
    void *src_row = in + y * pitch_in;

    OUT_T *dst_g = ((OUT_T *) dst_g_row) + x;
    OUT_T *dst_b = ((OUT_T *) dst_b_row) + x;
    OUT_T *dst_r = ((OUT_T *) dst_r_row) + x;

    uint16_t *src = ((uint16_t *) src_row) + 4 * x;

    *dst_r = *src++ >> bit_shift;
    *dst_g = *src++ >> bit_shift;
    *dst_b = *src++ >> bit_shift;
    if constexpr (has_alpha){
        void * dst_a_row = frame->data[3] + frame->linesize[3] * y;
        OUT_T *dst_a =((OUT_T *) dst_a_row) + x;
        *dst_a = *src >> bit_shift;
    }
}

template<typename FUNC>
__device__ void write_from_r12l(const uint8_t *src, FUNC WRITE_RES){

    comp_type_t r, g, b;

    r = (src[BYTE_SWAP(1)] & 0xFU) << 12U | src[BYTE_SWAP(0)] << 4U;                        //0
    g = src[BYTE_SWAP(2)] << 8U | (src[BYTE_SWAP(1)] & 0xF0U);
    b = (src[4 + BYTE_SWAP(0)] & 0xFU) << 12U | src[BYTE_SWAP(3)] << 4U;
    WRITE_RES(r, g, b);
    r = src[4 + BYTE_SWAP(1)] << 8U | (src[4 + BYTE_SWAP(0)] & 0xF0U);                      //1
    g = (src[4 + BYTE_SWAP(3)] & 0xFU) << 12U | (src[4 + BYTE_SWAP(2)]) << 4U;
    b = src[8 + BYTE_SWAP(0)] << 8U | (src[4 + BYTE_SWAP(3)] & 0xF0U);
    WRITE_RES(r, g, b);
    r = (src[8 + BYTE_SWAP(2)] & 0xFU) << 12U |src[8 + BYTE_SWAP(1)] << 4U;                 //2
    g = src[8 + BYTE_SWAP(3)] << 8U | (src[8 + BYTE_SWAP(2)] & 0xF0U);
    b = (src[12 + BYTE_SWAP(1)] & 0xFU) << 12U | src[12 + BYTE_SWAP(0)] << 4U;
    WRITE_RES(r, g, b);
    r = src[12 + BYTE_SWAP(2)] << 8U | (src[12 + BYTE_SWAP(1)] & 0xF0U);                    //3
    g = (src[16 + BYTE_SWAP(0)] & 0xFU) << 12U | src[12 + BYTE_SWAP(3)] << 4U;
    b = src[16 + BYTE_SWAP(1)] << 8U | (src[16 + BYTE_SWAP(0)] & 0xF0U);
    WRITE_RES(r, g, b);
    r = (src[16 + BYTE_SWAP(3)] & 0xFU) << 12U | src[16 + BYTE_SWAP(2)] << 4U;              //4
    g = src[20 + BYTE_SWAP(0)] << 8U | (src[16 + BYTE_SWAP(3)] & 0xF0U);
    b = (src[20 + BYTE_SWAP(2)] & 0xFU) << 12U | src[20 + BYTE_SWAP(1)] << 4U;
    WRITE_RES(r, g, b);
    r = src[20 + BYTE_SWAP(3)] << 8U | (src[20 + BYTE_SWAP(2)] & 0xF0U);                    //5
    g = (src[24 + BYTE_SWAP(1)] & 0xFU) << 12U | src[24 + BYTE_SWAP(0)] << 4U;
    b = src[24 + BYTE_SWAP(2)] << 8U | (src[24 + BYTE_SWAP(1)] & 0xF0U);
    WRITE_RES(r, g, b);
    r = (src[28 + BYTE_SWAP(0)] & 0xFU) << 12U | src[24 + BYTE_SWAP(3)] << 4U;              //6
    g = src[28 + BYTE_SWAP(1)] << 8U | (src[28 + BYTE_SWAP(0)] & 0xF0U);
    b = (src[28 + BYTE_SWAP(3)] & 0xFU) << 12U | src[28 + BYTE_SWAP(2)] << 4U;
    WRITE_RES(r, g, b);
    r = src[32 + BYTE_SWAP(0)] << 8U | (src[28 + BYTE_SWAP(3)] & 0xF0U);                    //7
    g = (src[32 + BYTE_SWAP(2)] & 0xFU) << 12U | src[32 + BYTE_SWAP(1)] << 4U;
    b = src[32 + BYTE_SWAP(3)] << 8U | (src[32 + BYTE_SWAP(2)] & 0xF0U);
    WRITE_RES(r, g, b);
}

template<typename FUNC>
__device__ void write_from_v210(const uint32_t *src, FUNC WRITE_RES){
    uint32_t w0_0 = *src++;
    uint32_t w0_1 = *src++;
    uint32_t w0_2 = *src++;
    uint32_t w0_3 = *src;
    uint16_t y, cb, cr;

    y = ((w0_0 >> 10U) & 0x3FFU) << 6U;
    cb = (w0_0 & 0x3FFU) << 6U;
    cr = ((w0_0 >> 20U) & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = (w0_1 & 0x3FFU) << 6U;
    cb = (w0_0 & 0x3FFU) << 6U;
    cr = ((w0_0 >> 20U) & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = ((w0_1 >> 20U) & 0x3FFU) << 6U;
    cb = ((w0_1 >> 10U) & 0x3FFU) << 6U;
    cr = (w0_2 & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = ((w0_2 >> 10U) & 0x3FFU) << 6U;
    cb = ((w0_1 >> 10U) & 0x3FFU) << 6U;
    cr = (w0_2 & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = (w0_3 & 0x3FFU) << 6U;
    cb = ((w0_2 >> 20U) & 0x3FFU) << 6U;
    cr = ((w0_3 >> 10U) & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = ((w0_3 >> 20U) & 0x3FFU) << 6U;
    cb = ((w0_2 >> 20U) & 0x3FFU) << 6U;
    cr = ((w0_3 >> 10U) & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);
}

template<typename IN_T, int bit_shift, bool has_alpha, AVPixelFormat CODEC>
__global__ void convert_rgb_from_inter(int width, int height, size_t pitch_in, char *in, AVFrame *out_frame){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_row = out_frame->data[0] + out_frame->linesize[0] *  y;
    void *src_row = in + y * pitch_in;

    IN_T *dst = ((IN_T *) dst_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *src = ((uint16_t *) src_row) + 4 * x;

    if constexpr (CODEC == AV_PIX_FMT_BGRA){
        *dst++ = src[2] >> bit_shift;//B
        *dst++ = src[1] >> bit_shift;//G
        *dst++ = src[0] >> bit_shift;//R
    } else if constexpr (CODEC == AV_PIX_FMT_X2RGB10LE){
        *dst++ = src[2] >> 6U; //B[->8]
        *dst++ = (((src[1] >> 6U) & 0x3FU) << 2U) | (src[2] >> 14U); // G[->6] | B[<-2]
        *dst++ = (((src[0] >> 6U) & 0xFU) << 4U) | (src[1] >> 12U); // R[->4] | G[<-4]
        *dst++ = (src[0] >> 10U); // [2x r<6]
    } else{
        *dst++ = *src++ >> bit_shift;
        *dst++ = *src++ >> bit_shift;
        *dst++ = *src++ >> bit_shift;
    }
    if constexpr (has_alpha && CODEC != AV_PIX_FMT_X2RGB10LE){
        *dst = *src >> bit_shift;
    }
}

template<bool has_alpha>
__global__ void convert_vuya_from_inter(int width, int height, size_t pitch_in, char *in, AVFrame *out_frame){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_row = out_frame->data[0] + out_frame->linesize[0] * y;
    void *src_row = in + y * pitch_in;

    uint8_t *dst = ((uint8_t *) dst_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *src = ((uint16_t *) src_row) + 4 * x;

    //uyva -> vuya
    *dst++ = src[2] >> 8U;
    *dst++ = src[0] >> 8U;
    *dst++ = src[1] >> 8U;
    if constexpr (has_alpha)
        *dst = src[3] >> 8U;
    else
        *dst = 0xFFU;
}

template<typename OUT_T,int bit_shift>
__global__ void convert_yuv422p_from_inter(int width, int height, size_t pitch_in, char *in, AVFrame *out_frame){
    // yuv 444 i -> yuvp
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width /2 || y >= height/2)
        return;

    OUT_T * dst_cb1, *dst_cb2, *dst_cr1, *dst_cr2, *dst_y1, *dst_y2;

    //y1, y2
    void * dst_y1_row = out_frame->data[0] + out_frame->linesize[0] * 2 * y;
    void * dst_y2_row = out_frame->data[0] + out_frame->linesize[0] * (2 * y + 1);
    dst_y1 = ((OUT_T *) (dst_y1_row)) + 2 * x;
    dst_y2 = ((OUT_T *) (dst_y2_row)) + 2 * x;

    //dst
    void * src1_row = in +  2 * y      * pitch_in;
    void * src2_row = in + (2 * y + 1) * pitch_in;
    uint16_t *src1 = ((uint16_t *) src1_row) + 4 * 2 * x;
    uint16_t *src2 = ((uint16_t *) src2_row) + 4 * 2 * x;

    //loops
    OUT_T *cb_loop[2];
    OUT_T *cr_loop[2];

    //fills different cr/cb for each line
    void * dst_cb1_row = out_frame->data[1] + out_frame->linesize[1] *  2 * y;
    void * dst_cb2_row = out_frame->data[1] + out_frame->linesize[1] *  (2 * y + 1);

    void * dst_cr1_row = out_frame->data[2] + out_frame->linesize[2] *  2 * y;
    void * dst_cr2_row = out_frame->data[2] + out_frame->linesize[2] *  (2 * y + 1);

    dst_cb1 = ((OUT_T *) (dst_cb1_row)) + x;
    dst_cb2 = ((OUT_T *) (dst_cb2_row)) + x;

    dst_cr1 = ((OUT_T *) (dst_cr1_row)) + x;
    dst_cr2 = ((OUT_T *) (dst_cr2_row)) + x;

    cb_loop[0] = dst_cb1;
    cb_loop[1] = dst_cb2;

    cr_loop[0] = dst_cr1;
    cr_loop[1] = dst_cr2;

    OUT_T *y_loop[2] = {dst_y1, dst_y2};
    uint16_t *src_loop[2] = {src1, src2};

    // each thread does 2 x 2 pixels
    for (int i = 0; i <2; ++i){
        uint16_t *src = src_loop[i];
        OUT_T *dst_y = y_loop[i];
        OUT_T *dst_cb = cb_loop[i];
        OUT_T *dst_cr = cr_loop[i];

        *dst_cb = ((src[0] / 2 + src[4] / 2) ) >> bit_shift; // U
        *dst_y++ = src[1] >> bit_shift; // Y
        *dst_cr = ((src[2] / 2 + src[6] / 2)) >> bit_shift; // V
        *dst_y = src[5] >> bit_shift; // Y
    }
}

template<typename OUT_T, int bit_shift>
__global__ void convert_yuv420p_from_inter(int width, int height, size_t pitch_in, char *in, AVFrame *out_frame){
    // yuv 444 i -> yuvp
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width /2 || y >= height/2)
        return;

    OUT_T * dst_cb1, *dst_cr1, *dst_y1, *dst_y2;

    //y1, y2
    void * dst_y1_row = out_frame->data[0] + out_frame->linesize[0] * 2 * y;
    void * dst_y2_row = out_frame->data[0] + out_frame->linesize[0] * (2 * y + 1);
    dst_y1 = ((OUT_T *) (dst_y1_row)) + 2 * x;
    dst_y2 = ((OUT_T *) (dst_y2_row)) + 2 * x;

    //dst
    void * src1_row = in +  2 * y      * pitch_in;
    void * src2_row = in + (2 * y + 1) * pitch_in;
    uint16_t *src1 = ((uint16_t *) src1_row) + 4 * 2 * x;
    uint16_t *src2 = ((uint16_t *) src2_row) + 4 * 2 * x;

    void * dst_cb_row = out_frame->data[1] + out_frame->linesize[1] *  y;
    void * dst_cr_row = out_frame->data[2] + out_frame->linesize[2] *  y;

    dst_cb1 = ((OUT_T *) (dst_cb_row)) + x;
    dst_cr1 = ((OUT_T *) (dst_cr_row)) + x;

    *dst_y1++ = src1[1] >> bit_shift; // Y
    *dst_y1 = src1[5] >> bit_shift; // Y

    *dst_y2++ = src2[1] >> bit_shift; // Y
    *dst_y2 = src2[5] >> bit_shift; // Y

    *dst_cb1 = ((src1[0] / 4 + src2[0] / 4 + src1[4] / 4 + src2[4] / 4) ) >> bit_shift; // U
    *dst_cr1 = ((src1[2] / 4 + src2[2]  /4 + src1[6] / 4 + src2[6] / 4) ) >> bit_shift; // V
}

template<typename IN_T, int bit_shift>
__global__ void convert_yuv444p_from_inter(int width, int height, size_t pitch_in, char * in, AVFrame *out_frame){
    //yuv444p -> yuv444i
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_y1_row = out_frame->data[0] + out_frame->linesize[0] *  y;
    void *dst_cb_row = out_frame->data[1] + out_frame->linesize[1] *  y;
    void *dst_cr_row = out_frame->data[2] + out_frame->linesize[2] *  y;
    void *src_row = in + y * pitch_in;

    IN_T   *dst_y1 = ((IN_T *) dst_y1_row) + x;
    IN_T   *dst_cb = ((IN_T *) dst_cb_row) + x;
    IN_T   *dst_cr = ((IN_T *) dst_cr_row) + x;
    uint16_t *src = ((uint16_t *) src_row) + 4 * x;

    *dst_cb = *src++ >> bit_shift; // U
    *dst_y1 = *src++ >> bit_shift; // Y
    *dst_cr = *src >> bit_shift; // V
}


template<typename OUT_T, int bit_shift>
__global__ void convert_p010le_from_inter(int width, int height, size_t pitch_in, char * in, AVFrame *out_frame)
{
    // y cbcr -> yuv 444 i
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height / 2)
        return;
    OUT_T * dst_cbcr, *dst_y1, *dst_y2;

    //y1, y2
    void * dst_y1_row = out_frame->data[0] + out_frame->linesize[0] * 2 * y;
    void * dst_y2_row = out_frame->data[0] + out_frame->linesize[0] * (2 * y + 1);
    dst_y1 = ((OUT_T *) (dst_y1_row)) + 2 * x;
    dst_y2 = ((OUT_T *) (dst_y2_row)) + 2 * x;

    void *dst_cbcr_row = out_frame->data[1] + out_frame->linesize[1] * y;
    dst_cbcr = ((OUT_T *) dst_cbcr_row) + 2 * x;

    //src
    void *src1_row = in + (y * 2) * pitch_in;
    void *src2_row = in + (y * 2 +1) * pitch_in;

    uint16_t *src1 = ((uint16_t *) src1_row) + 4 * 2 * x;
    uint16_t *src2 = ((uint16_t *) src2_row) + 4 * 2 * x;

    // U
    *dst_cbcr++ = (((src1[0] + src2[0] + src1[4] + src2[4]) / 4) & (0x3ff << 6)) >> bit_shift;

    // Y
    *dst_y1++ = src1[1] >> bit_shift;
    *dst_y2++ = src2[1] >> bit_shift;

    // V
    *dst_cbcr = (((src1[2] + src2[2] + src1[6] + src2[6]) / 4) & (0x3ff << 6)) >> bit_shift;

    // Y
    *dst_y1 = src1[5] >> bit_shift;
    *dst_y2 = src2[5] >> bit_shift;
}


__global__ void convert_ayuv64_from_inter(int width, int height, size_t pitch_in, char * in, AVFrame *out_frame)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_row = out_frame->data[0] + out_frame->linesize[0] *  y;
    void *src_row = in + y * pitch_in;

    uint16_t * src = ((uint16_t *) src_row) + 4 * x;
    uint16_t * dst = ((uint16_t *) dst_row) + 4 * x;

    *dst++ = src[3];
    *dst++ = src[1];
    *dst++ = src[0];
    *dst = src[2];
}


__global__ void convert_y210_from_inter(int width, int height, size_t pitch_in, char * in, AVFrame *out_frame){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    void *dst_row = out_frame->data[0] + out_frame->linesize[0] *  y;
    void *src_row = in + y * pitch_in;

    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;
    uint16_t *src = ((uint16_t *) src_row) + 4 * 2 * x;

    *dst++ = src[1]; //y
    *dst++ = (src[0] + src[4]) / 2; //u
    *dst++ = src[5]; //y22
    *dst = (src[2] + src[6]) / 2; //v
}

__global__ void convert_p210_from_inter(int width, int height, size_t pitch_in, char * in, AVFrame *out_frame){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    void *dst_y_row = out_frame->data[0] + out_frame->linesize[0] * y;
    void *dst_cbcr_row = out_frame->data[1] + out_frame->linesize[1] * y;
    void *src_row = in + y * pitch_in;

    uint16_t *dst_cbcr = ((uint16_t *) dst_cbcr_row) + 2 * x;
    uint16_t *dst_y = ((uint16_t *) dst_y_row) + 2 * x;
    uint16_t *src = ((uint16_t *) src_row) + 2 * 4 * x;

    uint16_t cb, cr;
    //uyva -> y cbcr
    cb = (src[0] + src[4]) / 2;
    cr = (src[2] + src[6]) / 2;

    *dst_cbcr++ = cb;
    *dst_cbcr = cr;

    *dst_y++ = src[1];
    *dst_y = src[5];
}
/**************************************************************************************************************/
/*                                             KERNELS TO                                                     */
/**************************************************************************************************************/

template<typename IN_T, int bit_shift, bool has_alpha, codec_t CODEC>
__global__ void convert_rgb_to_yuv_inter(int width, int height, size_t pitch_in, size_t pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    IN_T *src = ((IN_T *) src_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    comp_type_t r, g, b, y1, u, v;
    if constexpr (CODEC == R10k){
        uint8_t byte1 = *src++;
        uint8_t byte2 = *src++;
        uint8_t byte3 = *src++;
        uint8_t byte4 = *src++;

        r = byte1 << 8U | (byte2 & 0xC0U);
        g = (byte2 & 0x3FU) << 10U | (byte3 & 0xF0U) << 2U;
        b = (byte3 & 0xFU) << 12U | (byte4 & 0xFCU) << 4U;
        
    } else if (CODEC == BGR){
        r = src[2] << bit_shift;
        g = src[1] << bit_shift;
        b = src[0] << bit_shift;
    } else {
        r = *src++ << bit_shift;
        g = *src++ << bit_shift;
        b = *src++ << bit_shift;
    }

    u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);
    y1 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<12);
    v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);

    *dst++ = CLAMP_LIMITED_CBCR(u, 16);
    *dst++ = CLAMP_LIMITED_Y(y1, 16);
    *dst++ = CLAMP_LIMITED_CBCR(v, 16);

    if constexpr (has_alpha){
        *dst = *src << bit_shift;
    } else{
        *dst = 0xFFFFU;
    }
}


template<typename IN_T, int bit_shift, bool has_alpha, codec_t CODEC>
__global__ void convert_rgb_to_rgb_inter(int width, int height, size_t pitch_in, size_t pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    IN_T *src = ((IN_T *) src_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    uint16_t r, g, b;
    if constexpr (CODEC == R10k){
        uint8_t byte1 = *src++;
        uint8_t byte2 = *src++;
        uint8_t byte3 = *src++;
        uint8_t byte4 = *src++;

        r = byte1 << 8U | (byte2 & 0xC0U);
        g = (byte2 & 0x3FU) << 10U | (byte3 & 0xF0U) << 2U;
        b = (byte3 & 0xFU) << 12U | (byte4 & 0xFCU) << 4U;
    } else if constexpr (CODEC == BGR){
        r = src[2] << bit_shift;
        g = src[1] << bit_shift;
        b = src[0] << bit_shift;
    } else {
        r = *src++ << bit_shift;
        g = *src++ << bit_shift;
        b = *src++ << bit_shift;
    }

    *dst++ = r;
    *dst++ = g;
    *dst++ = b;

    if constexpr (has_alpha){
        *dst = *src << bit_shift;
    } else{
        *dst = 0xFFFFU;
    }
}

__global__ void convert_y416_to_rgb_inter(int width, int height, size_t pitch_in, size_t pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    uint16_t *src = ((uint16_t *) src_row) + 4 * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    comp_type_t r, g, b, y1, u, v;

    u = *src++;
    y1 = *src++;
    v = *src++;

    u = u - (1<<15);
    y1 = Y_SCALE * (y1 - (1<<12));
    v = v - (1<<15);

    r = YCBCR_TO_R_709_SCALED(y1, u, v) >> COMP_BASE;
    g = YCBCR_TO_G_709_SCALED(y1, u, v) >> COMP_BASE;
    b = YCBCR_TO_B_709_SCALED(y1, u, v) >> COMP_BASE;

    *dst++ = CLAMP_FULL(r, 16);
    *dst++ = CLAMP_FULL(g, 16);
    *dst++ = CLAMP_FULL(b, 16);

    *dst = *src;
}

template<typename IN_T, int bit_shift, bool is_reversed>
__global__ void convert_uyvy_to_yuv_inter(int width, int height, size_t pitch_in, size_t pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    IN_T *src = ((IN_T *) src_row) + 4 * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * 2 * x;

    if constexpr (is_reversed){
        *dst++ =  src[1] << bit_shift; //U
        *dst++ =  src[0] << bit_shift; //Y1
        *dst++ =  src[3] << bit_shift; //V
        *dst++ =  0xFFFFU; //A

        *dst++ =  src[1] << bit_shift; //U
        *dst++ =  src[2] << bit_shift; //Y2
        *dst++ =  src[3] << bit_shift; //V
        *dst =  0xFFFFU; //A
    } else {
        *dst++ =  src[0] << bit_shift; //U
        *dst++ =  src[1] << bit_shift; //Y1
        *dst++ =  src[2] << bit_shift; //V
        *dst++ =  0xFFFFU; //A

        *dst++ =  src[0] << bit_shift; //U
        *dst++ =  src[3] << bit_shift; //Y2
        *dst++ =  src[2] << bit_shift; //V
        *dst =  0xFFFFU; //A
    }
}


template<typename IN_T, int bit_shift, bool is_reversed>
__global__ void convert_uyvy_to_rgb_inter(int width, int height, size_t pitch_in, size_t pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    IN_T *src = ((IN_T *) src_row) + 4 * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * 2 * x;

    comp_type_t y1, y2, u, v, r ,g, b;

    if constexpr (is_reversed){
        y1 =  src[0] << bit_shift;
        u =  src[1] << bit_shift;
        y2 =  src[2] << bit_shift;
        v =  src[3] << bit_shift;
    } else {
        u =  src[0] << bit_shift;
        y1 =  src[1] << bit_shift;
        v =  src[2] << bit_shift;
        y2 =  src[3] << bit_shift;
    }
    u = u - (1<<15);
    y1 = Y_SCALE * (y1 - (1<<12));
    v = v - (1<<15);
    y2 = Y_SCALE * (y2 - (1<<12));

    r = YCBCR_TO_R_709_SCALED(y1, u, v) >> COMP_BASE;
    g = YCBCR_TO_G_709_SCALED(y1, u, v) >> COMP_BASE;
    b = YCBCR_TO_B_709_SCALED(y1, u, v) >> COMP_BASE;

    *dst++ = CLAMP_FULL(r, 16);
    *dst++ = CLAMP_FULL(g, 16);
    *dst++ = CLAMP_FULL(b, 16);
    *dst++ = 0xFFFFU;

    r = YCBCR_TO_R_709_SCALED(y2, u, v) >> COMP_BASE;
    g = YCBCR_TO_G_709_SCALED(y2, u, v) >> COMP_BASE;
    b = YCBCR_TO_B_709_SCALED(y2, u, v) >> COMP_BASE;

    *dst++ = CLAMP_FULL(r, 16);
    *dst++ = CLAMP_FULL(g, 16);
    *dst++ = CLAMP_FULL(b, 16);
    *dst = 0xFFFFU;
}


__global__ void convert_r12l_to_yuv_inter(int width, int height, size_t pitch_in, size_t pitch_out, const char * in, char *out)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 8 || y >= height)
        return;

    void *dst_row = out + pitch_out * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 8 * 4 * x;

    const void * src_row = in + pitch_in * y;
    const uint8_t *src = ((uint8_t *) src_row) + 36 * x;

    auto WRITE_RES = [dst](comp_type_t &r, comp_type_t &g, comp_type_t &b) mutable {
        comp_type_t y1, u, v;
        y1 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<12);
        u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);
        v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);

        *dst++ = CLAMP_LIMITED_CBCR(u, 16);
        *dst++ = CLAMP_LIMITED_Y(y1, 16);
        *dst++ = CLAMP_LIMITED_CBCR(v, 16);
        *dst++ = 0xFFFFU;
    };
    write_from_r12l(src, WRITE_RES);
}


__global__ void convert_r12l_to_rgb_inter(int width, int height, size_t pitch_in, size_t pitch_out, const char * in, char *out)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 8 || y >= height)
        return;

    void *dst_row = out + pitch_out * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 8 * 4 * x;

    const void * src_row = in + pitch_in * y;
    const uint8_t *src = ((uint8_t *) src_row) + 36 * x;

    auto WRITE_RES = [dst](uint16_t r, uint16_t g, uint16_t b) mutable {
        *dst++ = r;
        *dst++ = g;
        *dst++ = b;
        *dst++ = 0xFFFFU;
    };
    write_from_r12l(src, WRITE_RES);
}

__global__ void convert_v210_to_rgb_inter(int width, int height, size_t pitch_in, size_t pitch_out, const char * in, char *out)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 6 || y >= height)
        return;

    void *dst_row = out + pitch_out * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 6 * 4 * x;

    const void * src_row = in + pitch_in * y;
    const uint32_t *src = ((uint32_t *) src_row) + 4 * x;

    auto WRITE_RES = [dst](uint16_t &y11, uint16_t &cb1, uint16_t &cr1) mutable {
        comp_type_t y1, u, v, r, g, b;

        u = cb1 - (1<<15);
        y1 = Y_SCALE * (y11 - (1<<12));
        v = cr1 - (1<<15);

        r = YCBCR_TO_R_709_SCALED(y1, u, v) >> COMP_BASE;
        g = YCBCR_TO_G_709_SCALED(y1, u, v) >> COMP_BASE;
        b = YCBCR_TO_B_709_SCALED(y1, u, v) >> COMP_BASE;
        *dst++ = CLAMP_FULL(r, 16);
        *dst++ = CLAMP_FULL(g, 16);
        *dst++ = CLAMP_FULL(b, 16);
        *dst++ = 0xFFFFU;
    };

    write_from_v210(src, WRITE_RES);
}

__global__ void convert_v210_to_yuv_inter(int width, int height, size_t pitch_in, size_t pitch_out, const char * in, char *out)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 6 || y >= height)
        return;

    void *dst_row = out + pitch_out * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 6 * 4 * x;

    const void * src_row = in + pitch_in * y;
    const uint32_t *src = ((uint32_t *) src_row) + 4 * x;

    auto WRITE_RES = [dst](uint16_t &y1, uint16_t &cb, uint16_t &cr) mutable {
        *dst++ = cb;
        *dst++ = y1;
        *dst++ = cr;
        *dst++ = 0xFFFFU;
    };

    write_from_v210(src, WRITE_RES);
}


/**************************************************************************************************************/
/*                                               RGB TO                                                       */
/**************************************************************************************************************/

template<typename T, int shift, bool alpha, codec_t CODEC>
void rgb_to_yuv_inter(int width, int height, char *gpu_in_buffer, char *intermediate_to){

    size_t pitch_in = vc_get_linesize(width, CODEC);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_rgb_to_yuv_inter<T, shift, alpha, CODEC><<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

template<typename T, int shift, bool alpha, codec_t CODEC>
void rgb_to_rgb_inter(int width, int height, char *gpu_in_buffer, char *intermediate_to){

    size_t pitch_in = vc_get_linesize(width, CODEC);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_rgb_to_rgb_inter<T, shift, alpha, CODEC><<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void y416_to_rgb_inter(int width, int height, char *gpu_in_buffer, char *intermediate_to){

    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_y416_to_rgb_inter<<<grid, block>>>(width, height, pitch_out, pitch_out, gpu_in_buffer, intermediate_to);
}

template<typename T, int shift, bool is_reversed>
void uyvy_to_yuv_inter(int width, int height, char *gpu_in_buffer, char *intermediate_to){

    size_t pitch_in = vc_get_linesize(width, shift == 8 ? UYVY : Y216);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_uyvy_to_yuv_inter<T, shift, is_reversed><<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

template<typename T, int shift, bool is_reversed>
void uyvy_to_rgb_inter(int width, int height, char *gpu_in_buffer, char *intermediate_to){

    size_t pitch_in = vc_get_linesize(width, shift == 8 ? UYVY : Y216);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_uyvy_to_rgb_inter<T, shift, is_reversed><<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void r12l_to_yuv(int width, int height, char *gpu_in_buffer, char *intermediate_to){
    size_t pitch_in = vc_get_linesize(width, R12L);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_r12l_to_yuv_inter<<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void r12l_to_rgb(int width, int height, char *gpu_in_buffer, char *intermediate_to){
    size_t pitch_in = vc_get_linesize(width, R12L);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_r12l_to_rgb_inter<<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void v210_to_rgb(int width, int height, char *gpu_in_buffer, char *intermediate_to){
    size_t pitch_in = vc_get_linesize(width, v210);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 6 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_v210_to_rgb_inter<<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void v210_to_yuv(int width, int height, char *gpu_in_buffer, char *intermediate_to){
    size_t pitch_in = vc_get_linesize(width, v210);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 6 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_v210_to_yuv_inter<<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}
/**************************************************************************************************************/
/*                                              RGB FROM                                                      */
/**************************************************************************************************************/

template<typename T, int bit_shift, bool alpha>
void rgbp_from_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){
    size_t pitch_in = vc_get_linesize(width, Y416);
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_rgbp_from_inter<T, bit_shift, alpha><<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}

void ayuv64_from_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){
    size_t pitch_in = vc_get_linesize(width, Y416);
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_ayuv64_from_inter<<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}

template<typename T, int bit_shift, bool alpha, AVPixelFormat CODEC>
void rgb_from_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){
    size_t pitch_in = vc_get_linesize(width, Y416);
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_rgb_from_inter<T, bit_shift, alpha, CODEC><<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}

template<typename T, int bit_shift>
void yuv422p_from_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){
    size_t pitch_in = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    convert_yuv422p_from_inter<T, bit_shift><<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}

template<typename T, int bit_shift>
void yuv420p_from_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){
    size_t pitch_in = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    convert_yuv420p_from_inter<T, bit_shift><<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}

template<typename T, int bit_shift>
void yuv444p_from_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){
    size_t pitch_in = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    convert_yuv444p_from_inter<T, bit_shift><<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}

template<typename T, int bit_shift>
void p010le_from_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){
    size_t pitch_in = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    convert_p010le_from_inter<T, bit_shift><<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}

template<bool b>
void vuya_form_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){

    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_vuya_from_inter<b><<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}

void y210_form_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){

    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_y210_from_inter<<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}


void p210_from_inter(int width, int height, char *intermediate_to, AVFrame *dst_frame){

    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_p210_from_inter<<<grid, block>>>(width, height, pitch_in, intermediate_to, dst_frame);
}
/**************************************************************************************************************/
/*                                                LISTS                                                       */
/**************************************************************************************************************/

std::map<codec_t, void (*)(int, int, char *, char *)> conversions_to_rgb_inter = {
        {RGBA, rgb_to_rgb_inter<uint8_t, 8, true, RGBA>},
        {RGB, rgb_to_rgb_inter<uint8_t, 8, false, RGB>},
        {BGR, rgb_to_rgb_inter<uint8_t, 8, false, BGR>},
        {RG48, rgb_to_rgb_inter<uint16_t, 0, false, RG48>},
        {R10k, rgb_to_rgb_inter<char, 6, true, R10k>},

        {UYVY, uyvy_to_rgb_inter<uint8_t, 8, false>},
        {YUYV, uyvy_to_rgb_inter<uint8_t, 8, true>},
        {R12L, r12l_to_rgb},
        {v210, v210_to_rgb},
        {Y216, uyvy_to_rgb_inter<uint16_t, 0, true>}, //sus also
        {Y416, y416_to_rgb_inter},
};

std::map<codec_t, void (*)(int, int, char *, char *)> conversions_to_yuv_inter = {
        {RGBA, rgb_to_yuv_inter<uint8_t, 8, true, RGBA>},
        {RGB, rgb_to_yuv_inter<uint8_t, 8, false, RGB>},
        {BGR, rgb_to_yuv_inter<uint8_t, 8, false, BGR>},
        {RG48, rgb_to_yuv_inter<uint16_t, 0, false, RG48>},
        {R10k, rgb_to_yuv_inter<char, 6, true, R10k>},

        {UYVY, uyvy_to_yuv_inter<uint8_t, 8, false>},
        {YUYV, uyvy_to_yuv_inter<uint8_t, 8, false>},
        {R12L, r12l_to_yuv},
        {v210, v210_to_yuv},
        {Y216, uyvy_to_yuv_inter<uint16_t, 0, true>}, //y216 yuv420p sus colour
        {Y416, rgb_to_rgb_inter<uint16_t, 0, true, Y416>},
};

const std::map<AVPixelFormat, std::tuple<int, void (*)(int, int, char *, AVFrame *)>> conversions_from_inter = {
        // 10-bit YUV
        {AV_PIX_FMT_YUV420P10LE, {YUV_INTER_TO, yuv420p_from_inter<uint16_t, 6>}},
        {AV_PIX_FMT_YUV444P10LE, {YUV_INTER_TO, yuv444p_from_inter<uint16_t, 6>}},
        {AV_PIX_FMT_YUV422P10LE, {YUV_INTER_TO, yuv422p_from_inter<uint16_t, 6>}},
        {AV_PIX_FMT_P010LE, {YUV_INTER_TO, p010le_from_inter<uint16_t, 0>}},

        // 8-bit YUV (NV12)
        {AV_PIX_FMT_NV12, {YUV_INTER_TO, p010le_from_inter<uint8_t, 8>}},

        {AV_PIX_FMT_YUV420P, {YUV_INTER_TO, yuv420p_from_inter<uint8_t, 8>}},
        {AV_PIX_FMT_YUV422P, {YUV_INTER_TO, yuv422p_from_inter<uint8_t, 8>}},
        {AV_PIX_FMT_YUV444P, {YUV_INTER_TO, yuv444p_from_inter<uint8_t, 8>}},

        {AV_PIX_FMT_YUVJ420P, {YUV_INTER_TO, yuv420p_from_inter<uint8_t, 8>}},
        {AV_PIX_FMT_YUVJ422P, {YUV_INTER_TO, yuv422p_from_inter<uint8_t, 8>}},
        {AV_PIX_FMT_YUVJ444P, {YUV_INTER_TO, yuv444p_from_inter<uint8_t, 8>}},
        // 12-bit YUV
        {AV_PIX_FMT_YUV420P12LE, {YUV_INTER_TO, yuv420p_from_inter<uint16_t, 4>}},
        {AV_PIX_FMT_YUV422P12LE, {YUV_INTER_TO, yuv422p_from_inter<uint16_t, 4>}},
        {AV_PIX_FMT_YUV444P12LE, {YUV_INTER_TO, yuv444p_from_inter<uint16_t, 4>}},
        // 16-bit YUV
        {AV_PIX_FMT_YUV420P16LE, {YUV_INTER_TO, yuv420p_from_inter<uint16_t , 0>}},
        {AV_PIX_FMT_YUV422P16LE, {YUV_INTER_TO, yuv422p_from_inter<uint16_t , 0>}},
        {AV_PIX_FMT_YUV444P16LE, {YUV_INTER_TO, yuv444p_from_inter<uint16_t , 0>}},

        {AV_PIX_FMT_AYUV64LE, {YUV_INTER_TO, ayuv64_from_inter}},

        //GBRP
        {AV_PIX_FMT_GBRP, {RGB_INTER_TO, rgbp_from_inter<uint8_t, 8, false>}},
        {AV_PIX_FMT_GBRAP, {RGB_INTER_TO, rgbp_from_inter<uint8_t, 8, true>}},

        {AV_PIX_FMT_GBRP12LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 4, false>}},
        {AV_PIX_FMT_GBRP10LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 6, false>}},
        {AV_PIX_FMT_GBRP16LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 0, false>}},

        {AV_PIX_FMT_GBRAP12LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 4, true>}},
        {AV_PIX_FMT_GBRAP10LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 6, true>}},
        {AV_PIX_FMT_GBRAP16LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 0, true>}},

        //BGRA
        {AV_PIX_FMT_BGR0, {RGB_INTER_TO, rgb_from_inter<uint8_t, 8, true, AV_PIX_FMT_BGRA>}},
        {AV_PIX_FMT_BGRA, {RGB_INTER_TO, rgb_from_inter<uint8_t, 8, true, AV_PIX_FMT_BGRA>}},

        //RGB
        {AV_PIX_FMT_RGB24, {RGB_INTER_TO, rgb_from_inter<uint8_t, 8, false, AV_PIX_FMT_RGB24>}},
        {AV_PIX_FMT_RGB48LE, {RGB_INTER_TO, rgb_from_inter<uint16_t, 0, false, AV_PIX_FMT_RGB48LE>}},

        {AV_PIX_FMT_RGBA64LE, {RGB_INTER_TO, rgb_from_inter<uint16_t, 0, true, AV_PIX_FMT_RGBA64LE>}},
        {AV_PIX_FMT_RGBA, {RGB_INTER_TO, rgb_from_inter<uint8_t, 8, true, AV_PIX_FMT_RGBA>}},

#if Y210_PRESENT
        {AV_PIX_FMT_Y210, {YUV_INTER_TO, y210_form_inter}},
#endif
#if P210_PRESENT
        {AV_PIX_FMT_P210LE, {YUV_INTER_TO, p210_from_inter}},
#endif
#if XV3X_PRESENT
        {AV_PIX_FMT_XV30, {YUV_INTER_TO, y210_form_inter}}, //idk how to test these
        {AV_PIX_FMT_Y212, {YUV_INTER_TO, y210_form_inter}}, //idk how to test these
#endif
#if VUYX_PRESENT
        {AV_PIX_FMT_VUYA, {YUV_INTER_TO, vuya_from_inter<true>}}, //idk how to test these
        {AV_PIX_FMT_VUYX, {YUV_INTER_TO, vuya_from_inter<false>}}, //idk how to test these
#endif
#if X2RGB10LE_PRESENT
        {AV_PIX_FMT_X2RGB10LE, {RGB_INTER_TO, rgb_from_inter<uint8_t, 0, true, AV_PIX_FMT_X2RGB10LE>} }, //shift doesnt matter
#endif
};

/**************************************************************************************************************/
/*                                              INTERFACE                                                     */
/**************************************************************************************************************/

AVFrame *to_lavc_vid_conv_cuda(to_lavc_conv_cuda* state, const char *src) {
    auto internal_frame = state->frame;
    auto UG_codec = state->to;
    auto gpu_in_buffer = state->gpu_in_buffer;
    auto gpu_frame = state->gpu_frame;
    auto gpu_wrapper = state->gpu_wrapper;
    auto intermediate = state->intermediate_to;

    auto [inter, converter_from_inter] = conversions_from_inter.at(static_cast<AVPixelFormat>(internal_frame->format));

    //copy the image to gpu
    cudaMemcpy(gpu_in_buffer, src, vc_get_datalen(internal_frame->width, internal_frame->height, UG_codec), cudaMemcpyHostToDevice);
    //copy the destination to gpu
    cudaMemcpy(gpu_frame, &(gpu_wrapper->frame), sizeof(AVFrame), cudaMemcpyHostToDevice);

    if (inter == YUV_INTER_TO){
        auto converter_to = conversions_to_yuv_inter.at(UG_codec);
        converter_to(internal_frame->width, internal_frame->height, gpu_in_buffer, intermediate);
    } else if (inter == RGB_INTER_TO){
        auto converter_to = conversions_to_rgb_inter.at(UG_codec);
        converter_to(internal_frame->width, internal_frame->height, gpu_in_buffer, intermediate);
    } else {
        //error
    }

    converter_from_inter(internal_frame->width, internal_frame->height, intermediate, gpu_frame);

    //copy the converted image back to the host
    copy_to_host(gpu_wrapper, internal_frame);
    return internal_frame;
}

std::array subsamp = {
        AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUV420P10LE,
        AV_PIX_FMT_YUV420P12LE,
        AV_PIX_FMT_YUV420P16LE,
        AV_PIX_FMT_NV12,
        AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_P010LE,
};

void fill_AVFrmae(AVFrame *f){
    int q = std::find(subsamp.begin(), subsamp.end(), f->format) != subsamp.end() ? 2 : 1;
    for (int i = 0; i < 4; ++i)
        f->linesize[i] = av_image_get_linesize((AVPixelFormat) f->format, f->width, i);

    cudaMallocHost(&(f->data[0]), f->linesize[0] * f->height);

    for(int i = 1; i < 4; ++i)
        cudaMallocHost(&(f->data[i]), f->linesize[i] * f->height / q);
}

void free_host_AVFrame(AVFrame *f){
    for(int i = 0; i < 4; ++i)
        cudaFreeHost(f->data[i]);
}

to_lavc_conv_cuda *to_lavc_vid_conv_cuda_init(AVPixelFormat AV_codec, codec_t UG_codec, int width, int height){
    char *intermediate_to;
    char *gpu_in_buffer;
    AVFrame *gpu_frame;
    AVF_GPU_wrapper *gpu_wrapper;

    if (conversions_from_inter.find(AV_codec) == conversions_from_inter.end()
        || conversions_to_rgb_inter.find(UG_codec) == conversions_to_rgb_inter.end()){ //both should contain same keys
        return nullptr;
    }

    cudaMalloc(&intermediate_to, vc_get_datalen(width, height, Y416));
    cudaMalloc(&gpu_in_buffer, vc_get_datalen(width, height, UG_codec));
    cudaMalloc(&gpu_frame, sizeof(AVFrame));


    auto internal_frame = (AVFrame *) malloc(sizeof(AVFrame));
    internal_frame->width = width;
    internal_frame->height = height;
    internal_frame->format = AV_codec;
    fill_AVFrmae(internal_frame);

    gpu_wrapper = (AVF_GPU_wrapper *) malloc(sizeof(AVF_GPU_wrapper));
    alloc(gpu_wrapper, internal_frame);

    auto ret = (to_lavc_conv_cuda *) malloc(sizeof(to_lavc_conv_cuda));
    *ret = {internal_frame, UG_codec, intermediate_to, gpu_in_buffer, gpu_frame, gpu_wrapper };

    return ret;
}

void to_lavc_vid_conv_cuda_destroy(to_lavc_conv_cuda **s){
    auto state = *s;

    cudaFree(state->intermediate_to);
    cudaFree(state->gpu_in_buffer);
    cudaFree(state->gpu_frame);

    free_from_device(state->gpu_wrapper);
    delete state->gpu_wrapper;

    free_host_AVFrame(state->frame);
    free(state->frame);
    free(state);
    *s = nullptr;
}
