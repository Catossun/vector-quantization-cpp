#pragma once

std::deque <cv::Mat> separateMat2Square(const cv::Mat &mat, const int width);

std::deque <std::deque<int>> flatAllMat(const std::deque <cv::Mat> &mats);

std::deque<int> flatMat(const cv::Mat &mat);

void merge2SquareMat(const cv::Mat &dst, const std::deque <cv::Mat> &mats, int width);

std::deque <cv::Mat> vectors2Mats(const std::deque <std::deque<int>> &vectors, cv::Size matSize, int channels);

void vector2Mat(cv::Mat dst, const std::deque<int> &vector);

void testImage(int argc, char **argv);

void testExample();
