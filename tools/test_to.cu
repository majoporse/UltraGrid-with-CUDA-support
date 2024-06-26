#include "../src/libavcodec/from_lavc_vid_conv_cuda.h"
#include "../src/libavcodec/to_lavc_vid_conv_cuda.h"
#include <libavutil/pixfmt.h>
#include "../src/config_unix.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"

#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>
#include <ranges>
//extern "C" {
#include <libavutil/pixdesc.h>
//}

using std::chrono::milliseconds;
using namespace std::string_literals;



int main(int argc, char *argv[]){
    if (argc != 6){
        printf("bad input\n <width> <height> <in_name> <in_codec> <out_codec>\n");
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    codec_t UG_codec = get_codec_from_file_extension(argv[4]);
    AVPixelFormat AV_codec = av_get_pix_fmt(argv[5]);
    assert(AV_codec != AV_PIX_FMT_NONE && UG_codec != VIDEO_CODEC_NONE);

    std::ifstream fin(argv[3], std::ifstream::binary);
    std::ofstream fout1("AVtest_"s + argv[5] + ".rgb", std::ofstream::binary);
    std::ofstream reference("AVreference_"s + argv[5] + ".rgb", std::ofstream::binary);
    assert (width && height && fin && fout1 && reference);

    size_t in_size = vc_get_datalen(width, height, RGB);
    std::vector<unsigned char> fin_data(in_size);
    fin.read(reinterpret_cast<char *>(fin_data.data()), in_size);

    //RGB -> RG48 because it has conversion to every UG format
    std::vector<unsigned char> rg48vec(vc_get_datalen(width, height, RG48));
    auto d = get_decoder_from_to(RGB, RG48);
    for (int y = 0; y < height; ++y){
        d(rg48vec.data() + y * vc_get_linesize(width, RG48),
               fin_data.data()+ y * vc_get_linesize(width, RGB),
               vc_get_linesize(width, RG48), 0, 8, 16);
    }

    //rg48 -> ug codec
    auto decode = get_decoder_from_to(RG48, UG_codec);
    if (decode == NULL){
        std::cout << "cannot find RG48 -> UG format";
        return 1;
    }
    std::vector<unsigned char> UG_converted(vc_get_datalen(width, height, UG_codec));
    for (int y = 0; y < height; ++y){
        decode(UG_converted.data() + y * vc_get_linesize(width, UG_codec),
               rg48vec.data() + y * vc_get_linesize(width, RG48),
               vc_get_linesize(width, UG_codec), 0, 8, 16);
    }

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //convert UG -> AV
    //-------------------------------------------gpu version
    AVFrame *frame1 = nullptr;
    std::vector<char> dst_cpu1(vc_get_datalen(width, height, RGB));
    float count_gpu = 0;
    from_lavc_conv_state *from_state1;
    auto state = to_lavc_vid_conv_cuda_init(AV_codec, UG_codec, width, height);
    if (state->frame){
        for (int i = 0; i < 100; ++i){
            cudaEventRecord(start, 0);
            frame1 = to_lavc_vid_conv_cuda(state, reinterpret_cast<char *>(UG_converted.data()));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);
            count_gpu += time;
        }
        count_gpu /= 100.0;
        from_state1 = av_to_uv_conversion_cuda_init(frame1, RGB);
        if (from_state1){
            av_to_uv_convert_cuda(from_state1, frame1, dst_cpu1.data());
        }

    } else {
        std::cout << "non-existing gpu implementation\n";
    }

    //-------------------------------------------cpu version
    float count = 0;
    int max = 0;
    std::vector<char> dst_cpu2(vc_get_datalen(width, height, RGB));

    struct to_lavc_vid_conv *conv_to_av = to_lavc_vid_conv_init(UG_codec, width, height, AV_codec, 1);
    from_lavc_conv_state *from_state2;
    if (conv_to_av){
        AVFrame *frame2 = nullptr;
        for (int i = 0; i < 100; ++i){
            auto t1 = std::chrono::high_resolution_clock::now();
            frame2 = to_lavc_vid_conv(conv_to_av, reinterpret_cast<char *>(UG_converted.data())); //rg48->y210 segfault!!!
            auto t2 = std::chrono::high_resolution_clock::now();
            count += (t2-t1).count();
        }
        count /= 100.0;

        frame2->format = AV_codec; //these are not set inside the UG call
        frame2->width = width;
        frame2->height = height;
        from_state2 = av_to_uv_conversion_cuda_init(frame2, RGB);
        if (from_state2){

            av_to_uv_convert_cuda(from_state2, frame2, dst_cpu2.data());

            uint8_t *f1, *f2;
            f1 = (uint8_t *)dst_cpu1.data();
            f2 = (uint8_t *)dst_cpu2.data();
            for(int i = 0; i < vc_get_datalen(width, height, RGB); ++i) {
                max = std::max(std::abs( f1[i] - f2[i]), max);
            }
            //test validity against ug
            std::cout << "maximum difference against ultragrid implementation: " << max << "\n";

        }
    } else {
        std::cout << "non-existing cpu implementation\n";
    }

    //--------------------------------

    fout1.write(dst_cpu1.data(), vc_get_datalen(width, height, RGB));
    reference.write(dst_cpu2.data(), vc_get_datalen(width, height, RGB));

    //print time
    std::cout << "gpu implementation time: " << std::fixed  << std::setprecision(10) << count_gpu << "ms\n"
              << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1000'000.0<< "ms\n";
    std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

    av_to_uv_conversion_cuda_destroy(&from_state1);
    av_to_uv_conversion_cuda_destroy(&from_state2);
    to_lavc_vid_conv_cuda_destroy(&state);
}
