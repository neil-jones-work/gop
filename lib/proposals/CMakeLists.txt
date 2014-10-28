add_library( proposals saliency.cpp objectives.cpp unaryfeature.cpp unary.cpp edgefeature.cpp proposal.cpp geodesics.cpp seedfeature.cpp seed.cpp edgeweight.cpp )
target_link_libraries( proposals util learning imgproc contour lbfgs segmentation )

install(FILES edgefeature.h edgeweight.h geodesics.h objectives.h proposal.h saliency.h seed.h seedfeature.h unary.h unaryfeature.h DESTINATION /usr/local/include/we_cv_sdk/3rdparty/gop/proposals)
