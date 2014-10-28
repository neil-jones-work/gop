add_library( segmentation aggregation.cpp iouset.cpp segmentation.cpp )
target_link_libraries( segmentation util learning imgproc contour )

install(FILES aggregation.h iouset.h segmentation.h DESTINATION /usr/local/include/we_cv_sdk/3rdparty/gop/segmentation)
