#include "perceptron.hpp"
#include <numeric>

Perceptron::Perceptron(double learning_rate, int epochs)
    : lr(learning_rate), epochs(epochs), bias(0.0) {}

int Perceptron::activation(double x) const {
    return x >= 0 ? 1 : 0;
}

void Perceptron::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    if (X.empty()) return;
    
    size_t n_features = X[0].size();
    weights.assign(n_features, 0.0);
    bias = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            double linear_output = std::inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0) + bias;
            int y_predicted = activation(linear_output);
            
            double error = static_cast<double>(y[i]) - y_predicted;
            double update = lr * error;
            
            for (size_t j = 0; j < n_features; ++j) {
                weights[j] += update * X[i][j];
            }
            bias += update;
        }
    }
}

int Perceptron::predict(const std::vector<double>& x) const {
    double linear_output = std::inner_product(x.begin(), x.end(), weights.begin(), 0.0) + bias;
    return activation(linear_output);
}

std::vector<int> Perceptron::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    for (const auto& x : X) {
        predictions.push_back(predict(x));
    }
    return predictions;
}
