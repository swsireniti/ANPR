

SOURCES += main.cpp


QT += core gui widgets

CONFIG += c++17 console

CONFIG -= app_bundle


QMAKE_RPATHDIR += /usr/local/lib

INCLUDEPATH += /usr/local/include/opencv4

LIBS += -L/usr/local/lib

LIBS += -lopencv_core
LIBS += -lopencv_imgproc
LIBS += -lopencv_highgui
LIBS += -lopencv_imgcodecs
LIBS += -lopencv_dnn
LIBS += -lopencv_videoio



