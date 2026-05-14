#include <iostream>
#include <vector>
#include <iomanip>
#include "perceptron.hpp"
#include "gradient-descent.hpp"
#include "mlp.hpp"

void test_perceptron() {
    std::cout << "--- Perceptron (AND Problem) ---" << std::endl;
    Perceptron p(0.1, 10);
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int> y = {0, 0, 0, 1};
    
    p.fit(X, y);
    auto predictions = p.predict(X);
    
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] << "] "
                  << "Expected: " << y[i] << " Predicted: " << predictions[i] << std::endl;
    }
    std::cout << std::endl;
}

void test_gradient_descent() {
    std::cout << "--- Gradient Descent (Minimize y = x^2 + 1) ---" << std::endl;
    GradientDescent gd(0.1, 100);
    double start_x = 10.0;
    auto gradient = [](double x) { return 2 * x; };
    
    double min_x = gd.optimize(start_x, gradient);
    std::cout << "Minimum found at x = " << min_x << ", y = " << (min_x * min_x + 1) << std::endl;
    std::cout << std::endl;
}

void test_mlp() {
    std::cout << "--- MLP (XOR Problem) ---" << std::endl;
    MLP mlp(2, 4, 1, 0.1);
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};
    
    mlp.fit(X, y, 20000);
    auto predictions = mlp.predict(X);
    
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] << "] "
                  << "Expected: " << y[i][0] << " Predicted: " << predictions[i][0] << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    test_perceptron();
    test_gradient_descent();
    test_mlp();
    
    return 0;
}
