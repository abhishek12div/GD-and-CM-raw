import numpy as np
import matplotlib.pyplot as plt

#######################################################################################################################

def normalize(data):
    m1 = min(data)
    m2 = max(data)
    diff = m2 - m1
    y = data - m1
    a = -1
    b = 1
    X = a +((y*(b-a))/diff)
    return X

#train data
X_train = np.arange(-4,4,0.02)
#print(X_train)
Y_train = 1+(X_train + 2 * X_train**2)*np.sin(-X_train**2)

#normalize train data:
#leads to a better performance because gradient descent converges faster after normalization
X = normalize(X_train)
X = X.reshape(1, len(X))
#print(X)
Y = normalize(Y_train)
Y = Y.reshape(1, len(Y))

#print(X.shape, Y.shape)
#print(Y.shape[1])

#plt.figure()
#plt.plot(X, Y, '.')
#plt.show()

#######################################################################################################################

#test data
X_test = np.random.uniform(-4,4,10000)
Y_test = 1 + (X_test+2*X_test*X_test)*np.sin(-X_test*X_test)

X1= normalize(X_test)
X1 = X1.reshape(1, len(X1))
Y1 = normalize(Y_test)
Y1 = Y1.reshape(1, len(Y1))
print(X1.shape, Y1.shape)
#plt.figure()
#plt.plot(X1, Y1, '.')
#plt.show()

#######################################################################################################################

#Initialize neural network structure
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 7
    n_y = Y.shape[0]
    return n_x, n_h, n_y

#Initialize parameters
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.uniform(-0.5,0.5,[n_h, n_x])
    b1 = np.random.uniform(-0.5,0.5,[n_h, 1])
    W2 = np.random.uniform(-0.5,0.5,[n_y, n_h])
    b2 = np.random.uniform(-0.5,0.5,[n_y, 1])
    s1 = (n_h, n_x)
    s2 = (n_y, n_h)
    m_W1 = np.zeros(s1)
    m_b1 = np.zeros(s1)
    m_W2 = np.zeros(s2)
    m_b2 = np.zeros(1)


    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "m_W1": m_W1,
                  "m_b1": m_b1,
                  "m_W2": m_W2,
                  "m_b2": m_b2}
    return parameters

#Define Activation function
def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y

#Define derivative of the activation function: For backward propagation
def sigmoid_der(x):
    y = sigmoid(x)
    dy = y * (1 - y)
    return dy

#Forward Propagation
def fwd_prop(X, parameters): #X=X_train_norm
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    z1 = np.dot(W1, X) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = z2
    assert (a2.shape == (1, X.shape[1]))

    cache = {"z1": z1,
             "a1": a1,
             "z2": z2,
             "a2": a2}
    return a2, cache #cache is given as input during backpropagation

#Now, calculate the error difference between the obtained outputs and the Y_train_norm
def cost_func(a2, Y): #Y = Y_train_norm
    m = Y.shape[1] #=400
    err = np.mean(0.5*(a2-Y)**2)
    #err = 0.5/m * (np.sum((a2 - Y)**2))
    return err

#Backpropagate the Error
def back_prop(parameters, cache, X, Y):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    a1 = cache["a1"]
    z1 = cache["z1"]
    a2 = cache["a2"]
    z2 = cache["z2"]

    da2 = (a2 - Y)
    dz2 = da2 #* sigmoid_der(z2)
    dW2 = 1/m * np.dot(dz2, a1.T)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(W2.T, dz2) * sigmoid_der(z1)
    dW1 = 1/m * np.dot(dz1, X.T)
    db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

#update params w(k+1) = w(k) - alpha * dW(k)
def update_params(parameters, grads, u= 0.8, alpha=0.01):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    m_W1 = parameters["m_W1"]
    m_b1 = parameters["m_b1"]
    m_W2 = parameters["m_W2"]
    m_b2 = parameters["m_b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    m_W1 = u * m_W1 + dW1
    m_b1 = u * m_b1 + db1
    m_W2 = u * m_W2 + dW2
    m_b2 = u * m_b2 + db2

    W1 = W1 - alpha * m_W1
    b1 = b1 - alpha * m_b1
    W2 = W2 - alpha * m_W2
    b2 = b2 - alpha * m_b2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "m_W1": m_W1,
                  "m_b1": m_b1,
                  "m_W2": m_W2,
                  "m_b2": m_b2
                  }
    return parameters


def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    loss_list = []
    for i in range(0, num_iterations):
        a2, cache = fwd_prop(X, parameters)

        cost = cost_func(a2, Y)

        grads = back_prop(parameters, cache, X, Y)

        parameters = update_params(parameters, grads)

        loss_list.append(cost)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    markers = {'error'}
    x = np.arange(len(loss_list))
    plt.loglog(x, loss_list, label='error')

    plt.xlabel("trial_number")
    plt.ylabel("loss")
    plt.plot(0, np.max(loss_list))
    plt.legend(loc='upper right')

    return parameters

parameters = nn_model(X, Y, 7, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


W1 = parameters["W1"]
W2 = parameters["W2"]
b1 = parameters["b1"]
b2 = parameters["b2"]
x, y, z = layer_sizes(X1, Y1)

a2, cache = fwd_prop(X1, parameters)
error = cost_func(a2, Y1)
print(error)
plt.show()
