file(GLOB PRETRAINED_MODELS *.dat)
install(FILES ${PRETRAINED_MODELS} DESTINATION share/we-cv-sdk/pretrained_model/gop/)
