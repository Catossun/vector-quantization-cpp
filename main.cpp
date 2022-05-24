#include <iostream>
#include <deque>
#include <format>
#include <opencv2/opencv.hpp>
#include "VectorQuantizer.h"

#define PRINT_TITLE(title) {std::cout << "\n========== " << title << " ==========" << std::endl;}
#define PRINT_FOOTER(footer) {std::cout << "========== " << footer << " ==========" << std::endl;}

static const int ESC_KEY = 27;

std::deque<cv::Mat> separateMat2Square(const cv::Mat &mat, const int width);

std::deque<std::deque<int>> flatAllMat(const std::deque<cv::Mat> &mats);

std::deque<int> flatMat(const cv::Mat &mat);

void merge2SquareMat(const cv::Mat &dst, const std::deque<cv::Mat> &mats, int width);

std::deque<cv::Mat> vectors2Mats(const std::deque<std::deque<int>> &vectors, cv::Size matSize, int channels);

void vector2Mat(cv::Mat dst, const std::deque<int> &vector);

void testImage(int argc, char **argv);

void testExample();

int main(int argc, char **argv) {
    PRINT_TITLE("Run testExample()")
    testExample();
    PRINT_TITLE("Run testImage()")
    testImage(argc, argv);
    return 0;
}

void testExample() {
    std::deque<std::deque<int>> data = {
            std::deque<int>{135, 55},
            std::deque<int>{145, 65},
            std::deque<int>{165, 70},
            std::deque<int>{170, 85},
            std::deque<int>{180, 95},
            std::deque<int>{185, 100},
            std::deque<int>{205, 105},
            std::deque<int>{210, 125},
    };

    // Print data
    PRINT_TITLE("Source data")
    for (std::deque<int> vector:data) {
        for (int d:vector) {
            std::cout << d << ", ";
        }
        std::cout << std::endl;
    }

    VectorQuantizer vq(2, 4);
    PRINT_TITLE("Start encoding")
    const std::deque<int> &encodedData = vq.encode(data, 0.05);

    // Print data
    PRINT_TITLE("Encode data")
    for (int data:encodedData) {
        std::cout << data << ", " << std::endl;
    }

    const std::deque<std::deque<int>> &decodedData = vq.decode(encodedData);

    // Print data
    PRINT_TITLE("Decode data")
    for (std::deque<int> vectors:decodedData) {
        for (int d:vectors) {
            std::cout << d << ", ";
        }
        std::cout << std::endl;
    }
}

void testImage(int argc, char **argv) {
    const cv::Mat &img = cv::imread(argv[1]);

    PRINT_TITLE("Show original image")
    cv::imshow("Origin image", img);
    cv::waitKey();

    PRINT_TITLE("Converting image to vectors...")
    int squareWidth = 4;
    const std::deque<cv::Mat> rois = separateMat2Square(img, squareWidth);
    const std::deque<std::deque<int>> vectors = flatAllMat(rois);
    PRINT_FOOTER("Converting finished")

    PRINT_TITLE("Initializing vector quantizer...")
    int codeBookSize = 32;
    VectorQuantizer vq(vectors[0].size(), codeBookSize);
    PRINT_FOOTER("Initializing finished")

    PRINT_TITLE("Start training code book & Encoding vectors...")
    time_t startTime = time(nullptr);
    double endTrainRate = 0.05;
    const std::deque<int> &indexes = vq.encode(vectors, endTrainRate);
    time_t spendTime = time(nullptr) - startTime;
    std::cout << std::format("Run time(sec): {}", spendTime) << std::endl;
    PRINT_FOOTER("Training & Encoding finished")

    PRINT_TITLE("Decoding vectors...")
    const std::deque<std::deque<int>> &decodeVectors = vq.decode(indexes);
    const std::deque<cv::Mat> newRois = vectors2Mats(
            decodeVectors,
            cv::Size(squareWidth, squareWidth),
            img.channels()
    );
    PRINT_FOOTER("Decoding finished")

    PRINT_TITLE("Merging image...")
    const cv::Mat decodeImg(img.rows, img.cols, img.type());
    merge2SquareMat(decodeImg, newRois, squareWidth);
    PRINT_FOOTER("Merging finished")

    PRINT_TITLE("Show compressed image")
    cv::imshow("Compressed", decodeImg);

    while (cv::waitKey(0) != ESC_KEY);
    cv::destroyAllWindows();
    PRINT_TITLE("End of program")
}

void merge2SquareMat(const cv::Mat &dst, const std::deque<cv::Mat> &mats, int width) {
    int rows = dst.rows / width;
    int cols = dst.cols / width;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int offset = x + y * cols;
            cv::Mat mat = mats[offset];
            cv::Mat roi = dst(cv::Rect(x * width, y * width, mat.cols, mat.rows));
            mat.copyTo(roi);
        }
    }
}

std::deque<cv::Mat> vectors2Mats(const std::deque<std::deque<int>> &vectors, cv::Size matSize, int channels) {
    std::deque<cv::Mat> mats;
    for (const std::deque<int> &vector : vectors) {
        cv::Mat mat(matSize.height, matSize.width, CV_8UC(channels));
        vector2Mat(mat, vector);
        mats.push_back(mat);
    }
    return mats;
}

void vector2Mat(cv::Mat dst, const std::deque<int> &vector) {
    for (int y = 0; y < dst.rows; ++y) {
        uchar *row = dst.ptr(y);
        for (int x = 0; x < dst.cols; ++x) {
            uchar *pixels = row + x * dst.channels();
            int offset = (x + y * dst.cols) * dst.channels();
            for (int i = 0; i < dst.channels(); ++i) {
                pixels[i] = vector[offset + i];
            }
        }
    }
}

std::deque<std::deque<int>> flatAllMat(const std::deque<cv::Mat> &mats) {
    std::deque<std::deque<int>> vectors;
    for (const cv::Mat &mat : mats) {
        std::deque<int> vector = flatMat(mat);
        vectors.push_back(vector);
    }
    return vectors;
}

std::deque<int> flatMat(const cv::Mat &mat) {
    std::deque<int> vector;
    for (int y = 0; y < mat.rows; ++y) {
        const uchar *row = mat.ptr(y);
        for (int x = 0; x < mat.cols; ++x) {
            const uchar *pixel = row + x * mat.channels();
            for (int c = 0; c < mat.channels(); ++c) {
                vector.push_back(pixel[c]);
            }
        }
    }
    return vector;
}

std::deque<cv::Mat> separateMat2Square(const cv::Mat &mat, const int width) {
    std::deque<cv::Mat> rois;
    int rows = mat.rows / width;
    int cols = mat.cols / width;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            cv::Mat roi = mat(cv::Rect(x * width, y * width, width, width));
            rois.push_back(roi);
        }
    }

    return rois;
}
