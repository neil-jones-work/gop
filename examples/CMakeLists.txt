find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
include_directories( ../lib ${OpenCV_INCLUDE_DIRS} )
add_executable( example example.cpp )
target_link_libraries( example imgproc proposals contour ${OpenCV_LIBS} )

add_executable( example_learned example_learned.cpp )
target_link_libraries( example_learned imgproc proposals contour )
