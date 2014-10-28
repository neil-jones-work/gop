add_library( contour directedsobel.cpp boundarydetector.cpp sketchtokens.cpp structuredforest.cpp )
target_link_libraries( contour util imgproc learning )


install(FILES structuredforest.h boundarydetector.h sketchtokens.h DESTINATION /usr/local/include/we_cv_sdk/3rdparty/gop/contour)
