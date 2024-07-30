import numpy as np
from itertools import product
from random import gauss

num_of_hidden = 8
epoches = 10000
LR = 0.3

# func: init_hidden_weight
#   initialize hidden weight matrix with values between 0 and 1 (gaussian distribution)
def init_hidden_weight():
    w = [[gauss(0,1) for i in range(5)] for j in range(num_of_hidden)]
    return np.array(w)

# func: init_output_weight
#   initialize output weight matrix with values between 0 and 1 (gaussian distribution)
def init_output_weight():
    w=[gauss(0,1) for i in range(num_of_hidden+1)]
    return np.array(w).reshape((1,num_of_hidden+1))

# func: calc_results
#   calculate result for a given input
def calc_results(h_w, o_w, x):
    h_a = h_w.dot(x)    # calculate hidden arguments
    input_o = sigmoid(h_a)  # calculate the input of output layer
    input_o1 = np.r_[np.reshape(np.ones(16), (1, 16)), input_o]
    o_a = o_w.dot(input_o1)     # calculate output layer
    result = sigmoid(o_a)       # calculate final result
    return input_o1, result     # return the result

# func: calc_MSE
#   calculate and return the MSE
def calc_MSE(y, t):
    return ((y - t.T) ** 2).mean(axis = None)

# func: sigmoid
#   calculate and return sigmoid
def sigmoid(m):
  return 1 / (1 + np.exp(-m))

# func: gradient_descent
#   perform backpropagation (using gradient descent)
def back_propagation(x_data, hidden_data, output_data, input_o1, y, t):
    delta_out = y*(1-y)*(y-t.T)
    output_data1 = output_data[:,1:]
    input_o2 = input_o1[1:,:]
    delta_hidden = (output_data1.T.dot(delta_out)) * (input_o2 * (1-input_o2))
    new_output_data = output_data - LR * (delta_out.dot(input_o1.T))
    new_hidden_data = hidden_data - LR * (delta_hidden.dot(x_data.T))
    return new_hidden_data, new_output_data

# func: get_truth_table
#
#   generate all truth data has 4 digits.
#   16 binaries will be generated.
def get_truth_table():
    truth_list = list(product((0, 1), repeat=4))
    truth_list = [list(item) for item in truth_list]
    
    for i in range(len(truth_list)):
        truth_list[i].append(not (truth_list[i][0] ^ truth_list[i][1] ^ truth_list[i][2] ^ truth_list[i][3]))

    truth_list = np.array(truth_list)
    truth_list1 = np.c_[np.ones(16), truth_list]
    return truth_list1[:, 0:5].T, truth_list1[:, 5].reshape((16, 1))

if __name__ == '__main__':
    # get inital learning data
    x_data, sample_output = get_truth_table()
    losses = np.zeros((100, epoches))

    # perform learning.
    for i in range(100):
        hidden_w = init_hidden_weight()
        output_w = init_output_weight()

        input_o, result = calc_results(hidden_w, output_w, x_data)
        for j in range(epoches):
            # calculate loss
            loss = calc_MSE(result,sample_output)
            losses[i, j] = loss

            # perform back propagation for improving neural network
            hidden_w, output_w = back_propagation(x_data, hidden_w, output_w, input_o, result, sample_output)
            input_o, result = calc_results(hidden_w, output_w, x_data)

    for i in range(199, epoches, 200):
        print("Iteration {}: MSE={:.6f}".format(i+1, losses.mean(axis = 0)[i]))

    print("Number of hidden nodes: {0}".format(num_of_hidden))
    print("LR: {0}".format(LR))
    print("Final min loss: " + str(losses.mean(axis = 0)[epoches - 1]))
    
    print("Output weights: ")
    print(output_w.reshape(1, num_of_hidden + 1))

    # test cases
    test_cases = [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    ]

    input_o, result = calc_results(hidden_w, output_w, x_data)

    print("\nInput\t\t\tExpected Result\t\tResult")

    for test_case in test_cases:
        result_idx = int("".join(str(x) for x in test_case), 2)
        print("{}\t\t{}\t\t\t{:.4f}".format(test_case, sample_output[result_idx], result[0][result_idx]))
