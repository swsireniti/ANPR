#include <QApplication>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
//#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <QDir>
#include <QFileInfoList>
#include <chrono>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>


#include <thread>


std::string recognizePlate(cv::Mat plate,
                           cv::dnn::Net& ocrNet)
{
    //const std::string alphabet ="0123456789ABCEHKMOPTXY";
    const std::string alphabet ="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    cv::Mat gray;

    cv::cvtColor(plate,
                 gray,
                 cv::COLOR_BGR2GRAY);

    cv::resize(gray,
               gray,
               cv::Size(128, 32));

    gray.convertTo(gray,
                   CV_32F,
                   1.0 / 255.0);

    gray = (gray - 0.5f) / 0.5f;

    cv::Mat blob =
        cv::dnn::blobFromImage(gray);

    ocrNet.setInput(blob);

    cv::Mat out = ocrNet.forward();

    // shape:
    // [32,1,24]

    int seqLen = out.size[0];
    int classes = out.size[2];

    std::string result;

    int lastIndex = 0;

    for(int t = 0; t < seqLen; t++)
    {
        float maxVal = -9999.0f;
        int maxIndex = 0;

        for(int c = 0; c < classes; c++)
        {
            float val =
                out.at<float>(t, 0, c);

            if(val > maxVal)
            {
                maxVal = val;
                maxIndex = c;
            }
        }

        // CTC decode
        if(maxIndex != 0 &&
            maxIndex != lastIndex)
        {
            result += alphabet[maxIndex - 1];
        }

        lastIndex = maxIndex;
    }

    return result;
}


std::vector<cv::Rect> detectPlates(cv::Mat& frame,cv::dnn::Net& net,float confThreshold = 0.8f,float nmsThreshold = 0.45f)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    if(frame.empty())
        return boxes;

    cv::Mat blob;

    cv::dnn::blobFromImage(frame,
                           blob,
                           1.0 / 255.0,
                           cv::Size(640,640),
                           cv::Scalar(),
                           true,
                           false);

    net.setInput(blob);

    std::vector<cv::Mat> outputs;

    net.forward(outputs,
                net.getUnconnectedOutLayersNames());

    cv::Mat output = outputs[0];
    /*
    std::cout << "dims = " << output.dims << std::endl;

    for(int i = 0; i < output.dims; i++)
    {
        std::cout << "size[" << i << "] = "
                  << output.size[i] << std::endl;
    }
*/
    float xFactor = frame.cols / 640.0f;
    float yFactor = frame.rows / 640.0f;

    cv::Mat out(output.size[1],
                output.size[2],
                CV_32F,
                output.ptr<float>());

    cv::transpose(out, out);

    for(int i = 0; i < out.rows; i++)
    {
        float* data = out.ptr<float>(i);

        float cx = data[0];
        float cy = data[1];
        float w  = data[2];
        float h  = data[3];

        float confidence = data[4];

        if(confidence < confThreshold)
            continue;

        int left   = int((cx - 0.5f * w) * xFactor);
        int top    = int((cy - 0.5f * h) * yFactor);
        int width  = int(w * xFactor);
        int height = int(h * yFactor);

        boxes.push_back(
            cv::Rect(left, top, width, height)
            );

        confidences.push_back(confidence);
    }

    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        confThreshold,
        nmsThreshold,
        indices
        );

    std::vector<cv::Rect> result;

    for(size_t i = 0; i < indices.size(); i++)
    {
        result.push_back(
            boxes[indices[i]]
            );
    }

    return result;
}














void runOCRFolder(const QString& folderPath,
                  cv::dnn::Net& ocrNet)
{
    QDir dir(folderPath);

    QStringList filters;
    filters << "*.jpg"
            << "*.jpeg"
            << "*.png"
            << "*.bmp";

    QFileInfoList files =
        dir.entryInfoList(filters,
                          QDir::Files);

    for(const QFileInfo& fileInfo : files)
    {
        QString fileName =
            fileInfo.fileName();

        QString fullPath =
            fileInfo.absoluteFilePath();

        cv::Mat plate =
            cv::imread(
                fullPath.toStdString()
                );

        if(plate.empty())
        {
            std::cout
                << "Image load error: "
                << fullPath.toStdString()
                << std::endl;

            continue;
        }

        auto t0 =
            std::chrono::high_resolution_clock::now();

        std::string text =
            recognizePlate(
                plate,
                ocrNet
                );

        auto t1 =
            std::chrono::high_resolution_clock::now();

        double ocrMs =
            std::chrono::duration<double,std::milli>(
                t1 - t0
                ).count();

        std::cout
            << fileName.toStdString()
            << " -> "
            << text
            << " [OCR="
            << ocrMs
            << " ms]"
            << std::endl;
    }
}


void runFolder(const QString& folderPath, cv::dnn::Net& yoloNet, cv::dnn::Net& ocrNet)
{
    QDir dir(folderPath);

    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";

    QFileInfoList files = dir.entryInfoList(filters, QDir::Files);

    for(const QFileInfo& fileInfo : files)
    {
        QString fileName = fileInfo.fileName();
        QString fullPath = fileInfo.absoluteFilePath();

        cv::Mat frame = cv::imread(fullPath.toStdString());

        if(frame.empty())
        {
            std::cout << "Image load error: " << fullPath.toStdString() << std::endl;
            continue;
        }

        auto totalStart = std::chrono::high_resolution_clock::now();

        auto yoloStart = std::chrono::high_resolution_clock::now();
        auto boxes = detectPlates(frame, yoloNet);
        auto yoloEnd = std::chrono::high_resolution_clock::now();

        std::cout << fileName.toStdString() << " boxes=" << boxes.size() << " -> ";

        if(boxes.empty())
        {
            std::cout << "NO_PLATE" << std::endl;
            continue;
        }

        double ocrMs = 0.0;

        for(size_t i = 0; i < boxes.size(); i++)
        {
            cv::Mat plate = frame(boxes[i]).clone();

            auto ocrStart = std::chrono::high_resolution_clock::now();

            std::string text = recognizePlate(plate, ocrNet);

            auto ocrEnd = std::chrono::high_resolution_clock::now();

            ocrMs += std::chrono::duration<double,std::milli>(ocrEnd - ocrStart).count();

            std::cout << text;

            if(i + 1 < boxes.size())
                std::cout << " ";
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();

        double yoloMs = std::chrono::duration<double,std::milli>(yoloEnd - yoloStart).count();
        double totalMs = std::chrono::duration<double,std::milli>(totalEnd - totalStart).count();

        std::cout << " [YOLO=" << yoloMs << " ms OCR=" << ocrMs << " ms TOTAL=" << totalMs << " ms FPS=" << (1000.0 / totalMs) << "]" << std::endl;
    }
}

void runVideo(const std::string& videoPath,
              cv::dnn::Net& yoloNet,
              cv::dnn::Net& ocrNet)
{
    cv::VideoCapture cap(videoPath);

    if(!cap.isOpened())
    {
        std::cout << "Video open error: "
                  << videoPath
                  << std::endl;
        return;
    }

    double videoFps =
        cap.get(cv::CAP_PROP_FPS);

    if(videoFps <= 0)
        videoFps = 25.0;

    int delay =
        static_cast<int>(1000.0 / videoFps);

    std::cout
        << "Video FPS: "
        << videoFps
        << std::endl;

    cv::Mat frame;

    int frameCounter = 0;

    std::set<std::string> savedPlates;
    std::map<std::string,int> plateCounter;

    while(cap.read(frame))
    {
        frameCounter++;

        auto totalStart =
            std::chrono::high_resolution_clock::now();

        auto boxes =
            detectPlates(frame, yoloNet);

        for(auto& box : boxes)
        {
            cv::Rect safeBox =
                box &
                cv::Rect(
                    0,
                    0,
                    frame.cols,
                    frame.rows
                    );

            if(safeBox.width <= 0 ||
                safeBox.height <= 0)
            {
                continue;
            }

            cv::Mat plate =
                frame(safeBox).clone();

            std::string text =
                recognizePlate(
                    plate,
                    ocrNet
                    );

            if(text.length() == 8)
            {
                plateCounter[text]++;

                std::cout
                    << text
                    << " count="
                    << plateCounter[text]
                    << std::endl;

                if(plateCounter[text] >= 3)
                {
                    if(savedPlates.find(text) ==
                        savedPlates.end())
                    {
                        savedPlates.insert(text);

                        cv::imwrite(
                            "captures/" +
                                text +
                                ".jpg",
                            plate
                            );

                        std::cout
                            << "Saved plate: "
                            << text
                            << std::endl;
                    }
                }
            }

            cv::rectangle(
                frame,
                safeBox,
                cv::Scalar(0,255,0),
                2
                );

            cv::putText(
                frame,
                text,
                cv::Point(
                    safeBox.x,
                    safeBox.y - 10
                    ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0,255,0),
                2
                );
        }

        auto totalEnd =
            std::chrono::high_resolution_clock::now();

        double totalMs =
            std::chrono::duration<double,std::milli>(
                totalEnd - totalStart
                ).count();

        double procFps =
            1000.0 / totalMs;

        cv::putText(
            frame,
            cv::format(
                "PROC FPS %.2f",
                procFps
                ),
            cv::Point(20,40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0,255,255),
            2
            );

        cv::imshow(
            "ANPR",
            frame
            );

        int key =
            cv::waitKey(delay);

        if(key == 27)
            break;
    }

    cap.release();

    cv::destroyAllWindows();
}




//
struct ParkingCar
{
    std::string plate;
    std::chrono::system_clock::time_point enterTime;
    std::chrono::system_clock::time_point lastSeenTime;
    double enterVideoSec = 0.0;
    double lastSeenVideoSec = 0.0;
    int seenCount = 0;
    bool confirmed = false;
};

std::string formatDateTime(std::chrono::system_clock::time_point timePoint)
{
    std::time_t time =
        std::chrono::system_clock::to_time_t(timePoint);

    std::tm tm =
        *std::localtime(&time);

    std::ostringstream ss;

    ss << std::put_time(
        &tm,
        "%Y-%m-%d %H:%M:%S"
        );

    return ss.str();
}

void appendParkingLog(const ParkingCar& car,
                      std::chrono::system_clock::time_point exitTime,
                      double durationSec)
{
    const std::string logPath =
        "parking_log.csv";

    bool newFile =
        !QFileInfo(QString::fromStdString(logPath)).exists();

    std::ofstream file(
        logPath,
        std::ios::app
        );

    if(!file.is_open())
    {
        std::cout
            << "parking_log.csv open error"
            << std::endl;
        return;
    }

    if(newFile)
    {
        file
            << "plate,enter_time,exit_time,duration_seconds,duration"
            << std::endl;
    }

    int sec =
        (int)durationSec;

    int hours =
        sec / 3600;

    int minutes =
        (sec % 3600) / 60;

    int seconds =
        sec % 60;

    char durBuf[64];

    sprintf(
        durBuf,
        "%dh %02dm %02ds",
        hours,
        minutes,
        seconds
        );

    file
        << car.plate
        << ","
        << formatDateTime(car.enterTime)
        << ","
        << formatDateTime(exitTime)
        << ","
        << sec
        << ","
        << durBuf
        << std::endl;
}
void drawParkingInfo(cv::Mat& frame,
                     const std::vector<std::string>& events,
                     int activeCount)
{
    int panelHeight = 110;
    int top = frame.rows - panelHeight;

    if(top < 0)
        return;

    cv::Mat overlay = frame.clone();

    cv::rectangle(
        overlay,
        cv::Rect(0, top, frame.cols, panelHeight),
        cv::Scalar(20, 20, 20),
        cv::FILLED
        );

    cv::addWeighted(
        overlay,
        0.70,
        frame,
        0.30,
        0.0,
        frame
        );

    cv::putText(
        frame,
        cv::format("PARKING LOG   ACTIVE %d", activeCount),
        cv::Point(20, top + 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0,255,255),
        2
        );

    int y = top + 65;
    int start = 0;

    if(events.size() > 2)
        start = (int)events.size() - 2;

    for(size_t i = start; i < events.size(); i++)
    {
        cv::putText(
            frame,
            events[i],
            cv::Point(20, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(255,255,255),
            2
            );

        y += 30;
    }
}

void runParkingCamera(const std::string& videoPath,
                      cv::dnn::Net& yoloNet,
                      cv::dnn::Net& ocrNet)
{

    bool running = true;
    cv::VideoCapture cap;

    while(true)
    {
        cap.open(videoPath);

        if(cap.isOpened())
        {
            std::cout
                << "Camera connected"
                << std::endl;
            break;


        }

        std::cout
            << "Camera offline. Retry in 5 sec..."
            << std::endl;

        std::this_thread::sleep_for(
            std::chrono::seconds(5));
    }

    bool liveStream =
        videoPath.rfind("rtsp://", 0) == 0 ||
        videoPath.rfind("http://", 0) == 0 ||
        videoPath.rfind("https://", 0) == 0;

    QDir().mkpath("captures");

    cv::Mat frame;

    int frameCounter = 0;

    const int processEveryFrame = 30;
    const int confirmCount = 3;
    const double exitTimeoutSec = 20.0;

    double fpsVideo =
        cap.get(cv::CAP_PROP_FPS);

    if(fpsVideo <= 0.0)
        fpsVideo = 30.0;

    int width =
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);

    int height =
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer;

    writer.open(
        "parking_demo.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        fpsVideo / processEveryFrame,
        cv::Size(width, height)
        );

    if(!writer.isOpened())
    {
        std::cout
            << "VideoWriter open error: parking_demo.mp4"
            << std::endl;
    }

    std::map<std::string, ParkingCar> activeCars;
    std::vector<std::string> parkingEvents;

    auto wallStart =
        std::chrono::system_clock::now();

    auto steadyStart =
        std::chrono::steady_clock::now();


std::cout << "Window created" << std::endl;


   while(true)
    {
        if(!cap.read(frame))
        {
            std::cout
                << "RTSP lost. Reconnect..."
                << std::endl;

            cap.release();

            while(true)
            {
                std::this_thread::sleep_for(
                    std::chrono::seconds(2));

                cap.open(videoPath);

                if(cap.isOpened())
                {
                    std::cout
                        << "Camera reconnected"
                        << std::endl;

                    break;
                }
            }

            continue;
        }
        frameCounter++;

        double videoSec =
            cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;

        std::chrono::system_clock::time_point frameTime;

        if(liveStream)
        {
            auto now =
                std::chrono::system_clock::now();

            auto steadyNow =
                std::chrono::steady_clock::now();

            frameTime = now;

            videoSec =
                std::chrono::duration<double>(
                    steadyNow - steadyStart
                    ).count();
        }
        else
        {
            frameTime =
                wallStart +
                std::chrono::milliseconds(
                    (long long)(videoSec * 1000.0)
                    );
        }


        if(frameCounter % processEveryFrame != 0)
            continue;

        auto totalStart =
            std::chrono::high_resolution_clock::now();

        auto boxes =
            detectPlates(frame, yoloNet);

        for(auto& box : boxes)
        {
            cv::Rect safeBox =
                box &
                cv::Rect(
                    0,
                    0,
                    frame.cols,
                    frame.rows
                    );

            if(safeBox.width <= 0 ||
                safeBox.height <= 0)
            {
                continue;
            }

            cv::Mat plate =
                frame(safeBox).clone();

            std::string text =
                recognizePlate(
                    plate,
                    ocrNet
                    );

            if(text.length() == 8)
            {
                auto it =
                    activeCars.find(text);

                if(it == activeCars.end())
                {
                    ParkingCar car;

                    car.plate = text;
                    car.enterTime = frameTime;
                    car.lastSeenTime = frameTime;
                    car.enterVideoSec = videoSec;
                    car.lastSeenVideoSec = videoSec;
                    car.seenCount = 1;

                    activeCars[text] = car;
                }
                else
                {
                    it->second.lastSeenTime = frameTime;
                    it->second.lastSeenVideoSec = videoSec;
                    it->second.seenCount++;

                    if(!it->second.confirmed &&
                        it->second.seenCount >= confirmCount)
                    {
                        it->second.confirmed = true;

                        cv::imwrite(
                            "captures/" +
                                text +
                                "_enter.jpg",
                            plate
                            );

                        std::cout
                            << "ENTER: "
                            << text
                            << " time="
                            << formatDateTime(
                                   it->second.enterTime
                                   )
                            << std::endl;
                    }
                }
            }

            cv::Scalar boxColor(255,255,0);

            cv::rectangle(
                frame,
                safeBox,
                boxColor,
                2
                );

            cv::putText(
                frame,
                text,
                cv::Point(
                    safeBox.x,
                    safeBox.y - 10
                    ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                boxColor,
                2
                );
        }

        for(auto it = activeCars.begin();
             it != activeCars.end(); )
        {
            double missingSec =
                videoSec - it->second.lastSeenVideoSec;

            if(it->second.confirmed &&
                missingSec >= exitTimeoutSec)
            {
                double durationSec =
                    it->second.lastSeenVideoSec -
                    it->second.enterVideoSec;

                std::cout
                    << "EXIT: "
                    << it->second.plate
                    << " time="
                    << formatDateTime(
                           it->second.lastSeenTime
                           )
                    << " duration="
                    << (int)durationSec
                    << " sec"
                    << std::endl;

                appendParkingLog(
                    it->second,
                    it->second.lastSeenTime,
                    durationSec
                    );

                std::string eventText =
                    it->second.plate +
                    "  " +
                    formatDateTime(it->second.enterTime) +
                    " -> " +
                    formatDateTime(it->second.lastSeenTime) +
                    "  " +
                    std::to_string((int)durationSec) +
                    " sec";

                parkingEvents.push_back(eventText);

                it = activeCars.erase(it);
            }
            else if(!it->second.confirmed &&
                     missingSec >= exitTimeoutSec)
            {
                it = activeCars.erase(it);
            }
            else
            {
                ++it;
            }
        }

        auto totalEnd =
            std::chrono::high_resolution_clock::now();

        double totalMs =
            std::chrono::duration<double,std::milli>(
                totalEnd - totalStart
                ).count();

        double fps =
            1000.0 / totalMs;



        int y = 120;

        for(auto &car : activeCars)
        {
            if(!car.second.confirmed)
                continue;

            int sec =
                (int)(videoSec -
                       car.second.enterVideoSec);

            int hours = sec / 3600;
            int minutes = (sec % 3600) / 60;
            int seconds = sec % 60;

            char timeBuf[64];

            sprintf(
                timeBuf,
                "%dh %02dm %02ds",
                hours,
                minutes,
                seconds
                );

            cv::putText(
                frame,
                car.first + "  " + timeBuf,
                cv::Point(45,y),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0,0,100),
                2
                );

            y += 30;
        }

        //cv::Point(20,40)   // FPS
          //  cv::Point(220,40)  // ACTIVE
          //  cv::Point(20,80)   // активные машины

        cv::putText(
            frame,
            cv::format("FPS %.2f", fps),
            cv::Point(45,30),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0,0,100),
            2
            );

        cv::putText(
            frame,
            cv::format("ACTIVE %d", (int)activeCars.size()),
            cv::Point(230,30),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0,0,100),
            2
            );

        drawParkingInfo(
            frame,
            parkingEvents,
            (int)activeCars.size()
            );

        if(writer.isOpened())
        {
            writer.write(frame);
        }






        static bool windowInit = false;

        if(!windowInit)
        {
            cv::namedWindow("PARKING", cv::WINDOW_NORMAL);
            cv::resizeWindow("PARKING", 1024, 576); // размер окна
            windowInit = true;
        }

        cv::imshow("PARKING", frame);

       // if(cv::getWindowProperty(
        //        "PARKING",
         //       cv::WND_PROP_VISIBLE) < 1)
       // {
        //    break;
       // }
       // cv::imshow("PARKING", frame);

int key = cv::waitKey(1);

if(key == 27)
{
    running = false;
    break;
}










       // int key = cv::waitKey(1);

       // if(key == 27)
         //   break;
    }

    for(auto& item : activeCars)
    {
        ParkingCar& car =
            item.second;

        if(!car.confirmed)
            continue;

        double durationSec =
            car.lastSeenVideoSec -
            car.enterVideoSec;

        std::cout
            << "EXIT_END_VIDEO: "
            << car.plate
            << " time="
            << formatDateTime(car.lastSeenTime)
            << " duration="
            << (int)durationSec
            << " sec"
            << std::endl;

        appendParkingLog(
            car,
            car.lastSeenTime,
            durationSec
            );
    }

    cap.release();

    if(writer.isOpened())
    {
        writer.release();
    }

    cv::destroyAllWindows();
}


//





int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    std::cout << CV_VERSION << std::endl;
    // Загрузка YOLO
    std::cout
        << "CUDA devices: "
        << cv::cuda::getCudaEnabledDeviceCount()
        << std::endl;



    cv::dnn::Net net =cv::dnn::readNetFromONNX("model/best.onnx");//cv::dnn::readNetFromONNX("/home/pi/yolo/best.onnx");crnn
    if(net.empty())
    {
        std::cout << "YOLO load error" << std::endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget( cv::dnn::DNN_TARGET_CUDA);

    cv::dnn::Net ocrNet =cv::dnn::readNetFromONNX("model/crnn_ua_113.onnx");//crnn_ua2.onnx   crnn_ua_20.onnx  crnn2.onnx crnn_eu_20.onnx
    ocrNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    ocrNet.setPreferableTarget( cv::dnn::DNN_TARGET_CUDA);
    if(ocrNet.empty())
    {
        std::cout << "OCR load error" << std::endl;
    }
    else
    {
        std::cout << "OCR loaded" << std::endl;
    }
//"rtsp://admin:UNV264EA%40%2A@192.168.1.200:554/unicast/c1/s0/live"

    //runParkingCamera("rtsp://admin:UNV264EA%40%2A@192.168.1.200:554/unicast/c1/s1/live", net, ocrNet);
    //runParkingCamera("park_small.mp4", net, ocrNet);
    //runVideo("park_small.mp4", net, ocrNet);
    //runVideo("/media/pi/user/video/1_stable.mp4",net, ocrNet);


      //runParkingCamera("video/1_stable.mp4", net, ocrNet);  //1_stable.mp4
        runVideo("video/1_stable.mp4", net, ocrNet);

    //runFolder("img_10",net,ocrNet);

    //runFolder("/media/pi/user/autoriaNumberplateOcrEu-2023-06-26/test/img",net,ocrNet);
    //runFolder("/home/pi/yolo/img_10auto/",net,ocrNet);// 10 номеров с детекцией рамок  fps
    //runOCRFolder( "/home/pi/yolo/img_test/",ocrNet);
    //runOCRFolder( "/media/pi/user/autoriaNumberplateOcrEu-2023-06-26/test/img",ocrNet);// валидация вырезаных номеров
    // Картинка
    //cv::Mat frame = cv::imread("/home/pi/yolo/img/BK5831EX.jpg");//image.jpg  CA5500CM.jpg

    // if(frame.empty())
    // {
    //    std::cout << "Image load error" << std::endl;
    //    return -1;
    // }

    // Детекция
    //auto boxes = detectPlates(frame, net);

    //std::cout << "plates: " << boxes.size() << std::endl;

    // for(auto& box : boxes)
    // {
    //   cv::Mat plate =
    //       frame(box).clone();

    //   std::string text =
    //      recognizePlate(plate, ocrNet);

    //  std::cout << text << std::endl;

    // cv::rectangle(frame,
    //                box,
    //                cv::Scalar(0,255,0),
    //               2);

    // cv::putText(frame,
    //           text,
    //          cv::Point(box.x,
    //                   box.y - 10),
    //        cv::FONT_HERSHEY_SIMPLEX,
    //       1.0,
    //      cv::Scalar(0,255,0),
    //    2);
    // }

    //cv::namedWindow("result", cv::WINDOW_NORMAL);

    //cv::imshow("result", frame);

    // cv::waitKey(0);

    return 0;
}
