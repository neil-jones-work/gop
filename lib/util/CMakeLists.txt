add_library( util factory.cpp algorithm.cpp util.cpp eigen.cpp graph.cpp threading.cpp sse_defs.cpp optimization.cpp qp.cpp rasterize.cpp )
target_link_libraries( util lbfgs )

install(FILES algorithm.h eigen.h factory.h graph.h optimization.h qp.h rasterize.h sse_defs.h threading.h util.h win_util.h DESTINATION /usr/local/include/we_cv_sdk/3rdparty/gop/util)
