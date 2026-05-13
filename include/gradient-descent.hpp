#pragma once
#include <functional>
#include <vector>

class GradientDescent {
public:
    using Function = std::function<double(double)>;
    
    GradientDescent(double learning_rate = 0.01, int iterations = 100);
    
    double optimize(double x_start, const Function& gradient_func);

private:
    double lr;
    int n_iter;
};
