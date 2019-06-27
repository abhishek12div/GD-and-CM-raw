# Steepest-Gradient-Descent for Neural Network Training
Network Sturcture: 1-7-1 

Train data: 400 Samples 

Test data: 10000 Samples 

### Function used for train data(400 Samples):
#### y = f(x) = 1 + (X + 2 * X^2) * sin(-X^2) 
The same mapping function is used for test data, but with 10,000 samples
### Update Formula - SGD:
W(k+1) = W(k) + alpha * dW(k)

### Udate Formula - SGD+Momentum :
###### 1. Vector Update: m(k+1) = u * m(k) + dW(k)
###### 2. Parameter Update: W(k+1) = W(k) - alpha * m(k+1)

## Files:
#### 1. sgd.py - executes only the Steepest Gradient Update
#### 2. SGD+Momentum.py - executes only the Classical Momentum Update
#### 3. Combined-SGD+SGDM - When run, the graph clearly shows that Classical Momentum is Superior in minimizing the Error
#### 4. PPT file
## References:
Sebastian Ruder's Blog: http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms

Notes: http://cs231n.github.io/neural-networks-3/#sgd

Paper: http://cs229.stanford.edu/proj2015/054_report.pdf
