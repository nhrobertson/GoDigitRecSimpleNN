A simple neural network for recognizing digits using the MNIST digit data set.

Directly inspired/converted from Michael Nielsen's neural network and deep learning book: 
http://neuralnetworksanddeeplearning.com/index.html

Compare to network.py in the updated code repository for the book:
https://github.com/MichalDanielDobrzanski/DeepLearningPython/blob/master/network.py

Contains:

./data - Data directory containing the 4 MNIST data files 
    - test images and labels
    - training images and labels

go.mod - Go module file - includes the required packages using by    the network
    - Gonum - library for linear algebra - gonum.org/v1/gonum
    - MNIST - ease of access to the MNIST data - github.com/moverest/mnist

go.sum - "Used to validate the checksum of each direct and indirect dependency to confirm none of them has been modified" (https://golangbyexample.com/go-mod-sum-module/)

main.go - The main file containing the code for the neural network

    - network struct
        Structure containing:
        - numLayers - number of layers in the network
        - sizes - list of integers corresponding to the size of each layer, list indices must match the number of layers
        - weights - list of gonum Dense Matrices, the length of the list is the number of layers - 1 as the matrices is are the connections between layers
        - biases - list of gonum Dense Matrices, same as the biases

    - newNetwork function
        Function which creates and populates a network struct.
        Intakes number of layers and the list of sizes
        Returns a network
        -------------------------------------------------------------
        Defines the weights and biases as lists of *mat.Dense of size numLayers - 1

        Loop starts at index (i) of 1, meaning that it starts at the first hidden layer, not the input layer

        Each weight matrix is has rows of sizes[i] and cols of sizes[i-1], Ex. a network of [784, 10, 10] would have a weights list of [784x10, 10x10]

        Each bias matrix has rows of sizes[i] and cols of 1, Ex. a netowrk of [784, 10, 10] would have a biases list of [10, 10]

        Then random values are assigned to each matrix in each list a total of numLayers - 1 times (for i := 1; i < numLayers; i++)

        Returns a network struct of inputed numLayers, inputed sizes, random valued list of weights, and random valued list of biases.
        -------------------------------------------------------------

    - sigmoid function
        Function for calculating the sigmoid value of the input
        Inputs a float64 value z
        Outputs a float64 value
        -------------------------------------------------------------
        Returns 1.0/(1.0 + exp(z))
        -------------------------------------------------------------

    - sigmoidPrime function
        Function for calculating the derivative of the sigmoid function
        Intakes a float64 value z
        Returns a float64 value
        -------------------------------------------------------------
        Returns sigmoid(z) * (1 - sigmoid(z))
        -------------------------------------------------------------

    - cost_derivative function
        !!!! Only works for MNIST data set of output size 10 (0-9) !!!!
        Function for calculating the derivative of the cost function
        Intakes a matrix of the output of the network (output_a) and the label of the correct output (y)
        Outputs a matrix
        -------------------------------------------------------------
        Starts by converting the label into a usuable matrix for calculating the output.

        Creates a matrix of size 10x1 populated with 0s

        Sets index of the correct output value to 1.0

        Subtracts the output_a matrix by the y matrix

        Returns the result
        -------------------------------------------------------------

    - feedforward function
        Function for feeding forward through the network
        A function for struct network
        Intakes input matrix a
        Outputs the output matrix
        -------------------------------------------------------------
        Starts a loop for the length of the list of weights (numLayers - 1) 
        {

        Creates a z matrix = nn.weights[i] * a - nn.biases[i]

        Sets a to z

        Applies the sigmoid function to all the values in a

        }
        Returns a after the loop
        -------------------------------------------------------------

    - ShuffleTrainingData function
        Function for suffling the training_data
        Inputs a *mnist.Set of training_data
        -------------------------------------------------------------
        Gets the number of Images

        Creates lists of shuffledImages and Labels

        Shuffles the data using a rand.Perm(n)

        Sets the training_data Images and Labels to the shuffled data
        -------------------------------------------------------------

    - StochasiticGradientDescent function
        Function to start the training(learning) process using Stochastic Gradient Descent
        A function for struct network
        Intakes the size of the mini_batch, number of epochs, the learning rate eta, the training data, and test data
        -------------------------------------------------------------
        Gets the amount of training data and test data

        Starts a loop for the number of epochs
        {
        
        Shuffles the training data

        Creates a list of *mnist.Set to contain the mini_batches

        Starts a loop to populate the mini_batches list of mini_batchs of training data

        For each mini_batch in the list of mini_batch uses the update_mini_batch function to update the weights and biases

        Prints the which Epoch it is on the evaluates the network for each epoch
        }
        ------------------------------------------------------------

    - update_mini_batch function
        Function to continue the learning process by using a mini batch of training data to update the weights and biases
        A function for struct network
        Intakes a mini_batch and the learning rate eta
        -------------------------------------------------------------
        Creates nabla lists of weights and biases equal to the size of the weights and biases matrices set to 0s

        For the number of mini_batches 
        {
            Creates and sets delta nabla of weights and biases equal to the output of the backprop function

            Sets nabla b and w to: nabla + delta_nabla
        }

        Changes the weights and biases of the network by looping through each index in the matrix setting it equal to:
        the current value at the index - eta / (the size of the mini_batch) * nabla value
        -------------------------------------------------------------

    - backprop function
        Function to complete teh learning process using the backpropagation algorithm
        A function for struct network
        Inputs the input image and input label
        Outputs list of matrices (delta_nabla)
        -------------------------------------------------------------
        Creates new lists of nabla weights and biases populated with 0s

        Converts the image into a usuable matrix of number of pixels row and 1 column

        Creates a list of matrices for storing the z

        Creates a list of activation matrices and sets the first layers to the input layer a

        For the length of the list of weights
        {
        
        Feedforwards through the network by multipling the weights by the input a then adds the biases. Then applies sigmoid.

        Stores the updated a value in the next layer of the activations list

        }

        Starts backprop by setting delta equal to the cost_derivative function output with input of the last layer of the activation list and label y

        Multiplies that matrix by the sigmoidPrime version of the last layer of the zs matrix list

        Sets the last layer of output nabla_b to delta
        Sets the last layer of output nabla_w to delta * the last layer of activation list transposed.

        For the number of layers with index starting at 2
        {
            Sets a z matrix to zs list at (len(zs)-index)

            Sets a temporary matrix equal to z

            Applies sigmoidPrime to the temporary matrix

            Sets delta equal to the actual weights of the network at (len(weights)-l+1) transposed * delta

            Then multiples each element of delta by the temporary matrix

            Sets the output nabla_b at (len(nabla_b)-l) equal to delta

            Sets the output nabla_w at (len(nabla_w)-l) equal to delat * activations at (len(activations)-l-1) transposed
        }
        Returns the ouput nabla_b and nabla_w
        -------------------------------------------------------------

    - evaluate function
        Function for computing the correctness of the outputs using the feedforward function
        A function for struct network
        Inputs test_data
        Outputs int sum
        -------------------------------------------------------------
        Creates a list of test_results of size test_data

        For the number of test_data 
        {
        Creates a label matrix as before

        Creates a usable input layer using the test_data image

        Feedforwards the input layer

        Gets the output in a usable formate

        Sets the test result for that equal to that output
        }

        For the number of test results
        {
        Compares the test results to the label

        Adds 1 to the sum output if test_results[i] == test_data.Labels[i]
        }

        Returns sum
        -------------------------------------------------------------

    - main function
        Main function
        -------------------------------------------------------------
        Creates a network of sizes {784, 10, 10}

        Uses mnist package to set the training_data and test_data

        Starts StochasticGradientDescent for that network
        -------------------------------------------------------------

Bug List:

~~Training_data and Test_data not working as intended~~

~~Shuffle not working~~

~~Z values to large for sigmoid creating 0 matrices~~

~~Adjusting input image setting to 0~~

~~Delta in backprop being changed when applying sigmoid to temp: https://stackoverflow.com/questions/76695248/go-matrix-application-problem-in-backpropagation-algorithm~~

Adjusting of weights and biases causing de-learning instead of learning, Correct outputs going down for each epoch

