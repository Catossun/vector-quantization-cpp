#include <set>
#include <ctime>
#include <map>
#include <iostream>
#include <format>
#include "VectorQuantizer.h"

VectorQuantizer::VectorQuantizer(int dim, int codeBookSize) {
    this->dim = dim;
    this->codeBookSize = codeBookSize;
}

Vector VectorQuantizer::encode(std::deque<Vector> vectors, double endTrainRate) {
    generateCodeBook(vectors, endTrainRate);
    Vector encodedValue;
    for (Vector vector: vectors) {
        int minDistanceIndex = 0;
        int minDistance = INT32_MAX;
        for (int i = 0; i < codeBookSize; ++i) {
            Vector codeBookVector = codeBook[i];
            int distance = calculateDistance(codeBookVector, vector);
            if (distance < minDistance) {
                minDistance = distance;
                minDistanceIndex = i;
            }
        }
        encodedValue.push_back(minDistanceIndex);
    }
    return encodedValue;
}

void VectorQuantizer::generateCodeBook(std::deque<Vector> vectors, double endTrainRate) {
    codeBook = randomSelectInitVectors(vectors);
    trainCodeBook(vectors, endTrainRate);
}

void VectorQuantizer::trainCodeBook(std::deque<Vector> &vectors, double endTrainRate) {
    double trainRate = DBL_MAX;
    double lastAvgError = 0;
    while (true) {
        std::map<int, Vector> vectorIndexTable = classificationVectors(codeBook, vectors);
        double avgError = calculateAvgError(codeBook, vectors, vectorIndexTable);
        trainRate = abs((avgError - lastAvgError) / avgError);
        std::cout << std::format("[{}] Train rate: {}", ++trainTimes, trainRate) << std::endl;
        lastAvgError = avgError;

        if (trainRate <= endTrainRate)
            break;

        codeBook = calculateNewCodeBook(codeBook, vectors, vectorIndexTable);
    }
}

std::deque<Vector> VectorQuantizer::randomSelectInitVectors(std::deque<Vector> &vectors) {
    if (codeBookSize == vectors.size()) return vectors;

    std::deque<Vector> initVectors;

    std::set<int> pickedIndexes;
//    std::set<int> pickedIndexes = {0, 2, 3, 6}; // For example test
    srand(time(NULL));
    while (pickedIndexes.size() < codeBookSize) {
        pickedIndexes.insert(rand() % vectors.size());
    }

    for (int index : pickedIndexes) {
        initVectors.push_back(vectors[index]);
    }

    return initVectors;
}

std::map<int, Vector>
VectorQuantizer::classificationVectors(std::deque<Vector> &labelVectors,
                                       std::deque<Vector> &vectors) {
    std::map<int, Vector> indexTable;


    for (int i = 0; i < vectors.size(); ++i) {
        int minDistance = INT32_MAX;
        int minDistanceLabelIndex = -1;
        for (int j = 0; j < labelVectors.size(); ++j) {
            int distance = calculateDistance(vectors[i], labelVectors[j]);
            if (distance < minDistance) {
                minDistance = distance;
                minDistanceLabelIndex = j;
            }
        }
        if (indexTable.find(minDistanceLabelIndex) == indexTable.end()) {
            indexTable.insert(std::pair<int, Vector>(minDistanceLabelIndex, Vector()));
        }
        indexTable.at(minDistanceLabelIndex).push_back(i);
    }

    return indexTable;
}

int VectorQuantizer::calculateDistance(Vector &vectorA, Vector &vectorB) const {
    int distance = 0;
    for (int i = 0; i < vectorA.size(); ++i) {
        distance += pow(vectorA[i] - vectorB[i], 2);
    }
    return distance;
}

double
VectorQuantizer::calculateAvgError(std::deque<Vector> &labelVectors, std::deque<Vector> &vectors,
                                   std::map<int, Vector> &indexTable) {
    double avgError = 0;

    for (const std::pair<int, Vector> entity : indexTable) {
        for (const int index: entity.second) {
            avgError += calculateDistance(labelVectors[entity.first], vectors[index]);
        }
    }

    avgError /= vectors.size();

    return avgError;
}

std::deque<Vector>
VectorQuantizer::calculateNewCodeBook(std::deque<Vector> &labelVectors, std::deque<Vector> &vectors,
                                      std::map<int, Vector> &indexTable) {
    std::deque<Vector> newCodeBook;

    for (std::pair<int, Vector> entity:indexTable) {
        Vector newVector;
        for (int i = 0; i < dim; ++i) {
            int newValue = 0;
            for (int j = 0; j < entity.second.size(); ++j) {
                newValue += vectors[entity.second[j]][i];
            }
            newValue /= entity.second.size();
            newVector.push_back(newValue);
        }
        newCodeBook.push_back(newVector);
    }

    return newCodeBook;
}

std::deque<Vector> VectorQuantizer::decode(Vector indexes) {
    std::deque<Vector> decodedValue;

    for (const int index : indexes) {
        Vector vector = codeBook[index];
        decodedValue.push_back(vector);
    }

    return decodedValue;
}
