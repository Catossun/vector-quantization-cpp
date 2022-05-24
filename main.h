#pragma once

std::deque<cv::Mat> separateMat2Square(const cv::Mat &mat, int width);

std::deque<Vector> flatAllMat(const std::deque<cv::Mat> &mats);

Vector flatMat(const cv::Mat &mat);

void merge2SquareMat(const cv::Mat dst, const std::deque<cv::Mat> &mats, int width);

std::deque<cv::Mat> vectors2Mats(const std::deque<Vector> &vectors, cv::Size matSize, int channels);

void vector2Mat(cv::Mat dst, const Vector &vector);

void testImage(int argc, char **argv);

void testExample();
