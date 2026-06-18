#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <QDir>
#include <QFileInfoList>
#include <filesystem>
#include <set>
#include <map>



namespace fs = std::filesystem;
using namespace std;


std::string recognizePlate(cv::Mat plate,
                          cv::dnn::Net& ocrNet)
{
    //const std::string alphabet ="0123456789ABCEHKMOPTXY";
    const std::string alphabet ="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    cv::Mat gray;

    cv::cvtColor(plate,gray,cv::COLOR_BGR2GRAY);

    cv::resize(gray,gray,cv::Size(128, 32));

    gray.convertTo(gray,CV_32F,1.0 / 255.0);

    gray = (gray - 0.5f) / 0.5f;

    cv::Mat blob = cv::dnn::blobFromImage(gray);

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

    cv::dnn::blobFromImage(frame, blob,1.0 / 255.0,cv::Size(640,640),cv::Scalar(),true,false);

    net.setInput(blob);

    std::vector<cv::Mat> outputs;

    net.forward(outputs,net.getUnconnectedOutLayersNames());

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

    cv::Mat out(output.size[1], output.size[2], CV_32F,output.ptr<float>());

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

        boxes.push_back(cv::Rect(left, top, width, height));

        confidences.push_back(confidence);
    }

    std::vector<int> indices;

    cv::dnn::NMSBoxes(boxes,confidences,confThreshold,nmsThreshold,indices);

    std::vector<cv::Rect> result;

    for(size_t i = 0; i < indices.size(); i++)
    {
        result.push_back(boxes[indices[i]]);
    }

    return result;
}














void runOCRFolder(const QString& folderPath,
                  cv::dnn::Net& ocrNet)
{
    QDir dir(folderPath);

    QStringList filters;
    filters << "*.jpg"<< "*.jpeg" << "*.png"<< "*.bmp";

    QFileInfoList files =dir.entryInfoList(filters, QDir::Files);

    for(const QFileInfo& fileInfo : files)
    {
        QString fileName =fileInfo.fileName();

        QString fullPath =fileInfo.absoluteFilePath();

        cv::Mat plate =cv::imread(fullPath.toStdString());

        if(plate.empty())
        {
            std::cout<< "Image load error: "<< fullPath.toStdString()<< std::endl;

            continue;
        }

        auto t0 =std::chrono::high_resolution_clock::now();

        std::string text =recognizePlate(plate,ocrNet);

        auto t1 =std::chrono::high_resolution_clock::now();

        double ocrMs =std::chrono::duration<double,std::milli>(t1 - t0).count();

        std::cout<< fileName.toStdString() << " -> "<< text<< " [OCR="<< ocrMs<< " ms]"<< std::endl;
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

void runImage(const QString& imagePath, cv::dnn::Net& yoloNet, cv::dnn::Net& ocrNet)
{
    QFileInfo fileInfo(imagePath);

    QString fileName = fileInfo.fileName();
    QString fullPath = fileInfo.absoluteFilePath();

    cv::Mat frame = cv::imread(fullPath.toStdString());

    if(frame.empty())
    {
        std::cout << "Image load error: " << fullPath.toStdString() << std::endl;
        return;
    }

    auto totalStart = std::chrono::high_resolution_clock::now();

    auto yoloStart = std::chrono::high_resolution_clock::now();
    auto boxes = detectPlates(frame, yoloNet);
    auto yoloEnd = std::chrono::high_resolution_clock::now();

    std::cout << fileName.toStdString() << " boxes=" << boxes.size() << " -> ";

    if(boxes.empty())
    {
        std::cout << "NO_PLATE" << std::endl;

        cv::imshow("ANPR", frame);
        cv::waitKey(0);
        cv::destroyAllWindows();

        return;
    }

    double ocrMs = 0.0;

    for(size_t i = 0; i < boxes.size(); i++)
    {
        cv::Rect safeBox = boxes[i] & cv::Rect(0, 0, frame.cols, frame.rows);

        if(safeBox.width <= 0 || safeBox.height <= 0)
            continue;

        cv::Mat plate = frame(safeBox).clone();

        auto ocrStart = std::chrono::high_resolution_clock::now();

        std::string text = recognizePlate(plate, ocrNet);

        auto ocrEnd = std::chrono::high_resolution_clock::now();

        ocrMs += std::chrono::duration<double, std::milli>(ocrEnd - ocrStart).count();

        std::cout << text;

        if(i + 1 < boxes.size())
            std::cout << " ";

        cv::rectangle(frame, safeBox, cv::Scalar(255, 255, 0), 1);

        cv::putText(frame,text,cv::Point(safeBox.x, safeBox.y - 10),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255, 255, 0),1);
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();

    double yoloMs = std::chrono::duration<double, std::milli>(yoloEnd - yoloStart).count();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    double fps = 1000.0 / totalMs;

    std::cout << " [YOLO=" << yoloMs<< " ms OCR=" << ocrMs<< " ms TOTAL=" << totalMs<< " ms FPS=" << fps<< "]" << std::endl;

    cv::putText( frame,cv::format("FPS %.2f", fps),cv::Point(20, 40),cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(255, 255, 0),1);

    cv::imshow("ANPR", frame);

    cv::waitKey(0);

    cv::destroyAllWindows();
}

void runVideo(const std::string& videoPath,cv::dnn::Net& yoloNet,cv::dnn::Net& ocrNet)
{
    cv::VideoCapture cap(videoPath);
    // cv::VideoCapture cap;
    //cap.open(videoPath, cv::CAP_ANY);

    if(!cap.isOpened())
    {
        std::cout << "Video open error: "<< videoPath<< std::endl;
        return;
    }
    /*
        cv::VideoWriter writer;

        int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fpsVideo = cap.get(cv::CAP_PROP_FPS);

        writer.open(
            "result.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            fpsVideo,
            cv::Size(width, height)
        );

        if(!writer.isOpened())
        {
            std::cout << "VideoWriter open error" << std::endl;
            return;
        }

*/


    cv::Mat frame;

    int frameCounter = 0;

    std::set<std::string> savedPlates;


    std::map<std::string,int> plateCounter;
    std::filesystem::create_directories("captures");
    while(cap.read(frame))
    {
        frameCounter++;

        // можно пропускать кадры
        if(frameCounter % 3 != 0)
            continue;

        auto totalStart =std::chrono::high_resolution_clock::now();

        auto boxes =detectPlates(frame, yoloNet);

        for(auto& box : boxes)
        {
            cv::Rect safeBox = box &cv::Rect(0,0,frame.cols,frame.rows);

            if(safeBox.width <= 0 ||safeBox.height <= 0)
            {
                continue;
            }

            cv::Mat plate =frame(safeBox).clone();

            std::string text =recognizePlate( plate,ocrNet);

            //
            if(text.length() == 8)
            {
                plateCounter[text]++;
                std::cout<< text << " count="<< plateCounter[text]<< std::endl;

                if(plateCounter[text] >= 3)
                {
                    if(savedPlates.find(text) == savedPlates.end())
                    {
                        savedPlates.insert(text);

                        cv::imwrite("captures/" +text +".jpg",plate);

                        std::cout<< "Saved plate: "<< text<< std::endl;
                    }
                }
            }

            /*
            if(text.length() == 8)
            {
                if(savedPlates.find(text) ==
                   savedPlates.end())
                {
                    savedPlates.insert(text);

                    std::string fileName =
                        "/home/pi/yolo/captures/" +
                        text +
                        ".jpg";

                    cv::imwrite(
                        fileName,
                        plate
                        );

                    std::cout
                        << "Saved plate: "
                        << text
                        << std::endl;
                }
            }
            */
            //
            cv::rectangle(frame,safeBox,cv::Scalar(0,255,0),2);

            cv::putText(frame,text,cv::Point(safeBox.x,safeBox.y - 10 ),cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(0,255,0),2);
        }

        auto totalEnd =std::chrono::high_resolution_clock::now();

        double totalMs =std::chrono::duration<double,std::milli>(totalEnd - totalStart).count();

        double fps =1000.0 / totalMs;

        cv::putText(frame,cv::format("FPS %.2f", fps),cv::Point(20,40),cv::FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(0,255,255),2);

        //writer.write(frame);



        cv::imshow("ANPR", frame);

        int key = cv::waitKey(1);

        if(key == 27)
            break;
    }


    cap.release();
    //writer.release();

    cv::destroyAllWindows();






}






int main(int argc, char *argv[])
{
    //QApplication a(argc, argv);
    QCoreApplication a(argc, argv);
    std::cout << CV_VERSION << std::endl;

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    cv::dnn::Net net =cv::dnn::readNetFromONNX("model/best.onnx");

    if(net.empty())
    {
        std::cout << "YOLO load error" << std::endl;
        return -1;
    }

    net.setPreferableBackend(
        cv::dnn::DNN_BACKEND_OPENCV);

    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::dnn::Net ocrNet =cv::dnn::readNetFromONNX("model/crnn_ua_113.onnx");

    if(ocrNet.empty())
    {
        std::cout << "OCR load error" << std::endl;
        return -1;
    }

    ocrNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

    ocrNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::cout << "OCR loaded" << std::endl;


   // runFolder("C:/Users/SOTY_PC/Desktop/linux prim/ANPR-X86/img", net,ocrNet);

    //runVideo("C:/Users/SOTY_PC/Desktop/linux prim/ANPR-X86/Video/1_stable.mp4",net, ocrNet);
      runImage("img/BI3157CX.jpg", net, ocrNet);



    //runFolder( "C:/opencv_/img/",net,ocrNet);
//runImage("C:/opencv_/6.jpg", net, ocrNet);
    return 0;
}












