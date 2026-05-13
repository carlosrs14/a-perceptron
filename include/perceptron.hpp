#pragma once
#include <vector>

class Perceptron {
public:
    Perceptron(double learning_rate = 0.1, int epochs = 5);
    
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    int predict(const std::vector<double>& x) const;
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

private:
    double lr;
    int epochs;
    std::vector<double> weights;
    double bias;

    int activation(double x) const;
};
