#pragma once

#include <deque>

typedef std::deque<int> Vector;

class VectorQuantizer {
private:
    int codeBookSize = 0;
    int dim = 0;
    int trainTimes = 0;
    std::deque<Vector> codeBook;

    std::deque<Vector> randomSelectInitVectors(std::deque<Vector> &vectors);

    void generateCodeBook(std::deque<Vector> vectors, double endTrainRate);

    std::map<int, Vector>
    classificationVectors(std::deque<Vector> &labelVectors, std::deque<Vector> &vectors);

    double calculateAvgError(std::deque<Vector> &labelVectors, std::deque<Vector> &vectors,
                             std::map<int, Vector> &indexTable);

    int calculateDistance(Vector &vectorA, Vector &vectorB) const;

    std::deque<Vector>
    calculateNewCodeBook(std::deque<Vector> &labelVectors, std::deque<Vector> &vectors,
                         std::map<int, Vector> &indexTable);

    void trainCodeBook(std::deque<Vector> &vectors, double endTrainRate);

public:

    VectorQuantizer(int dim, int codeBookSize);

    Vector encode(std::deque<Vector> vectors, double endTrainRate);

    std::deque<Vector> decode(Vector indexes);
};
