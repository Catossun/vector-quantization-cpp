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

std::deque<int> VectorQuantizer::encode(std::deque<std::deque<int>> vectors, double endTrainRate) {
    generateCodeBook(vectors, endTrainRate);
    std::deque<int> encodedValue;
    for (std::deque<int> vector: vectors) {
        int minDistanceIndex = 0;
        int minDistance = INT32_MAX;
        for (int i = 0; i < codeBookSize; ++i) {
            std::deque<int> codeBookVector = codeBook[i];
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

void VectorQuantizer::generateCodeBook(std::deque<std::deque<int>> vectors, double endTrainRate) {
    codeBook = randomSelectInitVectors(vectors);
    trainCodeBook(vectors, endTrainRate);
}

void VectorQuantizer::trainCodeBook(std::deque<std::deque<int>> &vectors, double endTrainRate) {
    double trainRate = DBL_MAX;
    double lastAvgError = 0;
    while (true) {
        std::map<int, std::deque<int>> vectorIndexTable = classificationVectors(codeBook, vectors);
        double avgError = calculateAvgError(codeBook, vectors, vectorIndexTable);
        trainRate = abs((avgError - lastAvgError) / avgError);
        std::cout << std::format("[{}] Train rate: {}", ++trainTimes, trainRate) << std::endl;
        lastAvgError = avgError;

        if (trainRate <= endTrainRate)
            break;

        codeBook = calculateNewCodeBook(codeBook, vectors, vectorIndexTable);
    }
}

std::deque<std::deque<int>> VectorQuantizer::randomSelectInitVectors(std::deque<std::deque<int>> &vectors) {
    if (codeBookSize == vectors.size()) return vectors;

    std::deque<std::deque<int>> initVectors;

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

std::map<int, std::deque<int>>
VectorQuantizer::classificationVectors(std::deque<std::deque<int>> &labelVectors,
                                       std::deque<std::deque<int>> &vectors) {
    std::map<int, std::deque<int>> indexTable;


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
            indexTable.insert(std::pair<int, std::deque<int>>(minDistanceLabelIndex, std::deque<int>()));
        }
        indexTable.at(minDistanceLabelIndex).push_back(i);
    }

    return indexTable;
}

int VectorQuantizer::calculateDistance(std::deque<int> &vectorA, std::deque<int> &vectorB) const {
    int distance = 0;
    for (int i = 0; i < vectorA.size(); ++i) {
        distance += pow(vectorA[i] - vectorB[i], 2);
    }
    return distance;
}

double
VectorQuantizer::calculateAvgError(std::deque<std::deque<int>> &labelVectors, std::deque<std::deque<int>> &vectors,
                                   std::map<int, std::deque<int>> &indexTable) {
    double avgError = 0;

    for (const std::pair<int, std::deque<int>> entity : indexTable) {
        for (const int index: entity.second) {
            avgError += calculateDistance(labelVectors[entity.first], vectors[index]);
        }
    }

    avgError /= vectors.size();

    return avgError;
}

std::deque<std::deque<int>>
VectorQuantizer::calculateNewCodeBook(std::deque<std::deque<int>> &labelVectors, std::deque<std::deque<int>> &vectors,
                                      std::map<int, std::deque<int>> &indexTable) {
    std::deque<std::deque<int>> newCodeBook;

    for (std::pair<int, std::deque<int>> entity:indexTable) {
        std::deque<int> newVector;
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

std::deque<std::deque<int>> VectorQuantizer::decode(std::deque<int> indexes) {
    std::deque<std::deque<int>> decodedValue;

    for (const int index : indexes) {
        std::deque<int> vector = codeBook[index];
        decodedValue.push_back(vector);
    }

    return decodedValue;
}
