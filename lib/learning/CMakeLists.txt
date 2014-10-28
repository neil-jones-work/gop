add_library( learning splitcriterion.cpp treedata.cpp tree.cpp forest.cpp features.cpp )
target_link_libraries( learning util )

install(FILES features.h forest.h splitcriterion.h tree.h treedata.h DESTINATION /usr/local/include/we_cv_sdk/3rdparty/gop/learning)
