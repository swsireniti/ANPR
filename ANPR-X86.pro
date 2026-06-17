QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

INCLUDEPATH += F:\opencv\build\install\include
DEPENDPATH += F:\opencv\build\install\include

LIBS += -LF:\opencv\build\install\x64\vc17\lib \
        -lopencv_core4100d \
        -lopencv_imgproc4100d \
        -lopencv_highgui4100d \
        -lopencv_imgcodecs4100d \
        -lopencv_videoio4100d \
        -lopencv_video4100d \
        -lopencv_calib3d4100d \
        -lopencv_photo4100d \
        -lopencv_features2d4100d \
        -lopencv_objdetect4100d \
        -lopencv_flann4100d \
        -lopencv_dnn4100d

INCLUDEPATH += "F:/onnxruntime/include"
LIBS +=  F:\onnxruntime\lib\onnxruntime.lib

SOURCES += \
        main.cpp
