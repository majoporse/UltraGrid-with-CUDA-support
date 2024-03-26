FLAGS ?= -O2 -g
SRCDIR ?= ..
COMMON_FLAGS = $(FLAGS) -D_GNU_SOURCE -I$(SRCDIR)/src/
MKDIR_P = mkdir -p
CUDA_CXX = /usr/local/cuda/bin/nvcc
vpath %.c $(SRCDIR) $(SRCDIR)/tools
vpath %.cpp $(SRCDIR) $(SRCDIR)/tools
vpath %.cu $(SRCDIR) $(SRCDIR)/tools

TARGETS=astat_lib astat_test convert decklink_temperature uyvy2yuv422p thumbnailgen

all: $(TARGETS)

%.o : %.c
	$(MKDIR_P) $(dir $@)
	$(CC) -D HAVE_LAVC_CUDA_CONV=1 -D TEST_CONVERSIONS=1 $(COMMON_FLAGS) -c $< -o $@

%.o : %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D HAVE_LAVC_CUDA_CONV=1 -D TEST_CONVERSIONS=1 -std=c++17 $(COMMON_FLAGS) -c $< -o $@

%.o : %.cu
	$(MKDIR_P) $(dir $@)
	$(CUDA_CXX) -D HAVE_LAVC_CUDA_CONV=1 -D TEST_CONVERSIONS=1 -std c++20 $(COMMON_FLAGS) -c $< -o $@  -lavutil -lavcodec -lswscale


astat_test: astat.cpp src/compat/platform_pipe.o
	c++ -g -DASTAT_DEBUG -DDEFINE_TEST_MAIN $(COMMON_FLAGS) $^ -pthread -o astat_test

astat_lib: astat.a

astat.a: astat.o src/compat/platform_pipe.o
	ar rcs astat.a $^

convert: src/pixfmt_conv.o src/video_codec.o convert.o src/debug.o \
        src/utils/color_out.o src/utils/misc.o src/video_frame.o \
        src/utils/pam.c src/utils/y4m.c
	$(CXX) $^ -o convert

decklink_temperature: decklink_temperature.cpp ext-deps/DeckLink/Linux/DeckLinkAPIDispatch.o
	$(CXX) $^ -o $@

uyvy2yuv422p: uyvy2yuv422p.c
	$(CC) -g -std=c99 -Wall $< -o $@

thumbnailgen: thumbnailgen.o ipc_frame.o ipc_frame_unix.o
	$(CXX) $^ -o $@ -ljpeg

clean:
	$(RM) $(TARGETS) *o


benchmark_conv: benchmark_conv.o conv_utils.o src/debug.o src/libavcodec/from_lavc_vid_conv.o \
                                src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                src/video_frame.o
	$(CXX) $^ -o benchmark_conv -lavutil -lavcodec

benchmark: benchmark.o myconv.o myconv_inter.o src/debug.o src/libavcodec/from_lavc_vid_conv.o \
                                src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                src/video_frame.o
	$(CUDA_CXX) $^  -o benchmark -lavutil -lavcodec -lswscale

test_from: test_from.o src/debug.o src/libavcodec/from_lavc_vid_conv_cuda.o src/libavcodec/cuda_utils.o\
                                                       src/libavcodec/to_lavc_vid_conv_cuda.o src/libavcodec/from_lavc_vid_conv.o \
                                                       src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                                       src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                                       src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                                       src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                                       src/video_frame.o
	$(CUDA_CXX) $^  -o test_from  -lavutil -lavcodec -lswscale

test_to: test_to.o src/debug.o src/libavcodec/from_lavc_vid_conv_cuda.o src/libavcodec/cuda_utils.o\
                                                   src/libavcodec/to_lavc_vid_conv_cuda.o src/libavcodec/from_lavc_vid_conv.o \
                                                   src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                                   src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                                   src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                                   src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                                   src/video_frame.o
	$(CUDA_CXX) $^  -o test_to  -lavutil -lavcodec -lswscale

test_all_to: test_all_to.o src/debug.o src/libavcodec/from_lavc_vid_conv_cuda.o src/libavcodec/cuda_utils.o\
                                             src/libavcodec/to_lavc_vid_conv_cuda.o src/libavcodec/from_lavc_vid_conv.o \
                                             src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                             src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                             src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                             src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                             src/video_frame.o
	$(CUDA_CXX) $^  -o test_all_to  -lavutil -lavcodec -lswscale


test_all_from: test_all_from.o src/debug.o src/libavcodec/from_lavc_vid_conv_cuda.o src/libavcodec/cuda_utils.o\
                                src/libavcodec/to_lavc_vid_conv_cuda.o src/libavcodec/from_lavc_vid_conv.o \
                                src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                src/video_frame.o
	$(CUDA_CXX) $^  -o test_all_from  -lavutil -lavcodec -lswscale
