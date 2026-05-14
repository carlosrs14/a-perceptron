#include "mlp.hpp"
#include <cmath>
#include <random>
#include <numeric>

MLP::MLP(int input_size, int hidden_size, int output_size, double learning_rate)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size), lr(learning_rate) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0, 1);

    w1.resize(input_size, std::vector<double>(hidden_size));
    for (int i = 0; i < input_size; ++i)
        for (int j = 0; j < hidden_size; ++j)
            w1[i][j] = d(gen);

    b1.resize(hidden_size, 0.0);

    w2.resize(hidden_size, std::vector<double>(output_size));
    for (int i = 0; i < hidden_size; ++i)
        for (int j = 0; j < output_size; ++j)
            w2[i][j] = d(gen);

    b2.resize(output_size, 0.0);
}

double MLP::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double MLP::sigmoid_derivative(double x) const {
    return x * (1.0 - x);
}

void MLP::forward(const std::vector<double>& x) {
    z1.assign(hidden_size, 0.0);
    a1.assign(hidden_size, 0.0);
    for (int j = 0; j < hidden_size; ++j) {
        for (int i = 0; i < input_size; ++i) {
            z1[j] += x[i] * w1[i][j];
        }
        z1[j] += b1[j];
        a1[j] = sigmoid(z1[j]);
    }

    z2.assign(output_size, 0.0);
    a2.assign(output_size, 0.0);
    for (int j = 0; j < output_size; ++j) {
        for (int i = 0; i < hidden_size; ++i) {
            z2[j] += a1[i] * w2[i][j];
        }
        z2[j] += b2[j];
        a2[j] = sigmoid(z2[j]);
    }
}

void MLP::backward(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double> d_output(output_size);
    for (int i = 0; i < output_size; ++i) {
        double error = y[i] - a2[i];
        d_output[i] = error * sigmoid_derivative(a2[i]);
    }

    std::vector<double> d_hidden(hidden_size, 0.0);
    for (int i = 0; i < hidden_size; ++i) {
        double error_hidden = 0.0;
        for (int j = 0; j < output_size; ++j) {
            error_hidden += d_output[j] * w2[i][j];
        }
        d_hidden[i] = error_hidden * sigmoid_derivative(a1[i]);
    }

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            w2[i][j] += a1[i] * d_output[j] * lr;
        }
    }
    for (int j = 0; j < output_size; ++j) {
        b2[j] += d_output[j] * lr;
    }

    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            w1[i][j] += x[i] * d_hidden[j] * lr;
        }
    }
    for (int j = 0; j < hidden_size; ++j) {
        b1[j] += d_hidden[j] * lr;
    }
}

void MLP::fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            forward(X[i]);
            backward(X[i], y[i]);
        }
    }
}

std::vector<std::vector<double>> MLP::predict(const std::vector<std::vector<double>>& X) {
    std::vector<std::vector<double>> results;
    for (const auto& x : X) {
        forward(x);
        results.push_back(a2);
    }
    return results;
}
