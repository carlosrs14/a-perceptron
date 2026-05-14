#pragma once
#include <vector>

class MLP {
public:
    MLP(int input_size, int hidden_size, int output_size, double learning_rate = 0.1);
    
    void fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int epochs = 10000);
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& X);

private:
    int input_size, hidden_size, output_size;
    double lr;
    
    std::vector<std::vector<double>> w1, w2;
    std::vector<double> b1, b2;
    
    std::vector<double> a1, a2;
    std::vector<double> z1, z2;

    double sigmoid(double x) const;
    double sigmoid_derivative(double x) const;
    
    void forward(const std::vector<double>& x);
    void backward(const std::vector<double>& x, const std::vector<double>& y);
};

