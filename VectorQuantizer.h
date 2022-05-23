#pragma once

#include <deque>

class VectorQuantizer {
private:
    int codeBookSize = 0;
    int dim = 0;
    int trainTimes = 0;
    std::deque<std::deque<int>> codeBook;

    std::deque<std::deque<int>> randomSelectInitVectors(std::deque<std::deque<int>> &vectors);

    void generateCodeBook(std::deque<std::deque<int>> vectors, double endTrainRate);

    std::map<int, std::deque<int>>
    classificationVectors(std::deque<std::deque<int>> &labelVectors, std::deque<std::deque<int>> &vectors);

    double calculateAvgError(std::deque<std::deque<int>> &labelVectors, std::deque<std::deque<int>> &vectors,
                             std::map<int, std::deque<int>> &indexTable);

    int calculateDistance(std::deque<int> &vectorA, std::deque<int> &vectorB) const;

    std::deque<std::deque<int>>
    calculateNewCodeBook(std::deque<std::deque<int>> &labelVectors, std::deque<std::deque<int>> &vectors,
                         std::map<int, std::deque<int>> &indexTable);

    void trainCodeBook(std::deque<std::deque<int>> &vectors, double endTrainRate);

public:

    VectorQuantizer(int dim, int codeBookSize);

    std::deque<int> encode(std::deque<std::deque<int>> vectors, double endTrainRate);

    std::deque<std::deque<int>> decode(std::deque<int> indexes);
};
