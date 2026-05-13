#include "gradient-descent.hpp"

GradientDescent::GradientDescent(double learning_rate, int iterations)
    : lr(learning_rate), n_iter(iterations) {}

double GradientDescent::optimize(double x_start, const Function& gradient_func) {
    double x = x_start;
    for (int i = 0; i < n_iter; ++i) {
        double grad = gradient_func(x);
        x = x - lr * grad;
    }
    return x;
}
