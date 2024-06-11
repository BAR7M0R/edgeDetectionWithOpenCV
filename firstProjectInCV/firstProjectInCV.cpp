#include <opencv2/opencv.hpp> 
#include <iostream> 
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <array>

static void generateScharrImage(const cv::Mat source, const std::string name)
{
    cv::Mat X = cv::Mat::zeros(source.size(), source.type());
    cv::Mat Y = cv::Mat::zeros(source.size(), source.type());
    cv::Mat out = cv::Mat::zeros(source.size(), source.type());
    auto start = std::chrono::system_clock::now();
    cv::Scharr(source, X, -1, 1, 0);
    cv::Scharr(source, Y, -1, 0, 1);
    out = cv::abs(X) + cv::abs(Y);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    std::string label1(name);
    std::string label2("time: " + std::to_string(elapsed_seconds.count()));
    size_t x = 0;
    size_t y = 0;
    size_t w = label1.length() * 8;
    size_t h = 50;
    cv::putText(out, label1, cv::Point(x + int(w / 100), y + int(h / 2)), cv::FONT_HERSHEY_SIMPLEX, 7. / 15, cv::Scalar(255, 255, 255));
    cv::putText(out, label2, cv::Point(x + int(w / 100), y + int(h)), cv::FONT_HERSHEY_SIMPLEX, 7. / 15, cv::Scalar(255, 255, 255));

    cv::imshow(name, out);
    cv::imwrite("out\\" + name + ".jpg", out);
}
static void generateCannyImage(cv::Mat source, const std::string name)
{
    cv::Mat out = cv::Mat::zeros(source.size(), source.type());
    auto start = std::chrono::system_clock::now();
    cv::Canny(source, out, 100, 200);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    std::string label1(name);
    std::string label2("time: " + std::to_string(elapsed_seconds.count()));
    size_t x = 0;
    size_t y = 0;
    size_t w = label1.length() * 8;
    if (w > out.cols)
    {
        w = static_cast<size_t>(out.cols) - 1;
    }
    size_t h = 50;
    cv::putText(out, label1, cv::Point(x + int(w / 100), y + int(h / 2)), cv::FONT_HERSHEY_SIMPLEX, 7. / 15, cv::Scalar(255, 255, 255));
    cv::putText(out, label2, cv::Point(x + int(w / 100), y + int(h)), cv::FONT_HERSHEY_SIMPLEX, 7. / 15, cv::Scalar(255, 255, 255));
    cv::imshow(name, out);
    cv::imwrite("out\\" + name + ".jpg", out);
}
static void generateSobelImage(const cv::Mat source, const std::string name)
{
    cv::Mat X = cv::Mat::zeros(source.size(), source.type());
    cv::Mat Y = cv::Mat::zeros(source.size(), source.type());
    cv::Mat out = cv::Mat::zeros(source.size(), source.type());
    auto start = std::chrono::system_clock::now();
    cv::Sobel(source, X, -1, 1, 0);
    cv::Sobel(source, Y, -1, 0, 1);
    out = cv::abs(X) + cv::abs(Y);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    std::string label1(name);
    std::string label2("time: " + std::to_string(elapsed_seconds.count()));
    size_t x = 0;
    size_t y = 0;
    size_t w = label1.length() * 8;
    if (w > out.cols)
    {
        w = static_cast<size_t>(out.cols) - 1;
    }
    size_t h = 50;
    cv::putText(out, label1, cv::Point(x + int(w / 100), y + int(h / 2)), cv::FONT_HERSHEY_SIMPLEX, 7. / 15, cv::Scalar(255, 255, 255));
    cv::putText(out, label2, cv::Point(x + int(w / 100), y + int(h)), cv::FONT_HERSHEY_SIMPLEX, 7. / 15, cv::Scalar(255, 255, 255));
    cv::imshow(name, out);
    cv::imwrite("out\\" + name + ".jpg", out);
}
static cv::Mat myGrayscale(const cv::Mat image)
{
    auto grayValue = [](int r, int g, int b) {return static_cast<int>(0.7152 * r + 0.2126 * g + 0.0722 * b); };
    cv::Mat editedImage = cv::Mat::zeros(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            editedImage.at<cv::Vec3b>(y, x)[0] = grayValue(image.at<cv::Vec3b>(y, x)[2], image.at<cv::Vec3b>(y, x)[1], image.at<cv::Vec3b>(y, x)[0]);
            editedImage.at<cv::Vec3b>(y, x)[1] = grayValue(image.at<cv::Vec3b>(y, x)[2], image.at<cv::Vec3b>(y, x)[1], image.at<cv::Vec3b>(y, x)[0]);
            editedImage.at<cv::Vec3b>(y, x)[2] = grayValue(image.at<cv::Vec3b>(y, x)[2], image.at<cv::Vec3b>(y, x)[1], image.at<cv::Vec3b>(y, x)[0]);
        }
    }
    return editedImage;
}
static cv::Mat myGrayscale2(const cv::Mat image)
{
    auto grayValue = [](int r, int g, int b) {return static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b); };
    cv::Mat editedImage = cv::Mat::zeros(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            editedImage.at<cv::Vec3b>(y, x)[0] = grayValue(image.at<cv::Vec3b>(y, x)[2], image.at<cv::Vec3b>(y, x)[1], image.at<cv::Vec3b>(y, x)[0]);
            editedImage.at<cv::Vec3b>(y, x)[1] = grayValue(image.at<cv::Vec3b>(y, x)[2], image.at<cv::Vec3b>(y, x)[1], image.at<cv::Vec3b>(y, x)[0]);
            editedImage.at<cv::Vec3b>(y, x)[2] = grayValue(image.at<cv::Vec3b>(y, x)[2], image.at<cv::Vec3b>(y, x)[1], image.at<cv::Vec3b>(y, x)[0]);
        }
    }
    return editedImage;
}
int main(int argc, char** argv)
{
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    cv::Mat image = cv::imread("C:\\Users\\Studia\\Desktop\\WIN_20240608_12_43_24_Pro.jpg"); //set path into image here dont forget set double back-slesch
    cv::resize(image, image, cv::Size(), 0.25, 0.25);
    cv::Mat gaussianBlurPic;
    cv::Mat medianBlurPic;
    cv::Mat averageBlurPic;
    cv::Mat bilateralFiltePic;
    cv::Mat fastNonLocalMeansDenoisingPic;

    cv::Mat imagegray = myGrayscale(image);

    start = std::chrono::system_clock::now();
    cv::GaussianBlur(imagegray, gaussianBlurPic, cv::Size(5, 5), 0);    
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    //cv::putText(gaussianBlurPic, "gaussianBlurPic, time: " + std::to_string(elapsed_seconds.count()), cv::Point(20, 20), 1, 1, cv::Scalar(123)); //add this line if you want to print generation time
    
    start = std::chrono::system_clock::now();
    cv::medianBlur(imagegray, medianBlurPic, 5);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    //cv::putText(medianBlurPic, "medianBlurPic, time: " + std::to_string(elapsed_seconds.count()), cv::Point(20, 20), 1, 1, cv::Scalar(123));
    
    start = std::chrono::system_clock::now();
    cv::blur(imagegray, averageBlurPic, cv::Size(5, 5));
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    //cv::putText(averageBlurPic, "averageBlurPic, time: " + std::to_string(elapsed_seconds.count()), cv::Point(20, 20), 1, 1, cv::Scalar(123));
    
    start = std::chrono::system_clock::now();
    cv::bilateralFilter(imagegray, bilateralFiltePic, 9, 75, 75);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    //cv::putText(bilateralFiltePic, "bilateralFiltePic, time: " + std::to_string(elapsed_seconds.count()), cv::Point(20, 20), 1, 1, cv::Scalar(123));

    start = std::chrono::system_clock::now();
    cv::fastNlMeansDenoisingColored(imagegray, fastNonLocalMeansDenoisingPic, 3.0, 3.0, 7, 21);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    //cv::putText(fastNonLocalMeansDenoisingPic, "fastNonLocalMeansDenoisingPic, time: " + std::to_string(elapsed_seconds.count()), cv::Point(20, 20), 1, 1, cv::Scalar(123));
    ////////////////////////////////////////////////////


    generateCannyImage(gaussianBlurPic, "CannyEdgesGaussianBlurPic");
    generateCannyImage(medianBlurPic, "CannyEdgesMedianBlurPic");
    generateCannyImage(averageBlurPic, "CannyEdgesAverageBlurPic");
    generateCannyImage(bilateralFiltePic, "CannyEdgesBilateralFiltePic");
    generateCannyImage(fastNonLocalMeansDenoisingPic, "CannyEdgesFastNonLocalMeansDenoisingPic");

    generateSobelImage(gaussianBlurPic, "SobelEdgesgaussianBlurPic");
    generateSobelImage(medianBlurPic, "SobelEdgesMedianBlurPic");
    generateSobelImage(averageBlurPic, "SobelEdgesAverageBlurPic");
    generateSobelImage(bilateralFiltePic, "SobelEdgesBilateralFiltePic");
    generateSobelImage(fastNonLocalMeansDenoisingPic, "SobelEdgesFastNonLocalMeansDenoisingPic");

    generateScharrImage(gaussianBlurPic, "ScharrEdgesGaussianBlurPic");
    generateScharrImage(medianBlurPic, "ScharrEdgesmedianBlurPic");
    generateScharrImage(averageBlurPic, "ScharrEdgesAverageBlurPic");
    generateScharrImage(bilateralFiltePic, "ScharrEdgesBilateralFiltePic");
    generateScharrImage(fastNonLocalMeansDenoisingPic, "ScharrEdgesFastNonLocalMeansDenoisingPic");

    cv::waitKey(0);
    return 0;

}