find_package( JPEG REQUIRED )
find_package( PNG REQUIRED )

include_directories( ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIR} )

add_library( imgproc color.cpp filter.cpp gradient.cpp morph.cpp nms.cpp resample.cpp image.cpp )
target_link_libraries( imgproc util ${JPEG_LIBRARIES} ${PNG_LIBRARIES} )

install(FILES color.h filter.h gradient.h image.h morph.h nms.h resample.h DESTINATION /usr/local/include/we_cv_sdk/3rdparty/gop/imgproc)
