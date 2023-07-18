package main

import (
	//"encoding/csv"

	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/moverest/mnist"

	"gonum.org/v1/gonum/mat"
)

// Struct for the network
type network struct {
	numLayers int
	sizes     []int
	weights   []*mat.Dense
	biases    []*mat.Dense
}

// Generates a new network with a specified number of Layers and Sizes of those layers
// The weights and biases are then randomly assigned between a number 0-1
// Returns a network of: number numLayers, slice of sizes, a list of transposed weight matrices, and a list of bias vectors
func newNetwork(numLayers int, sizes []int) *network {
	//Creates a temporary storage for the edges between two layers
	weights := make([]*mat.Dense, numLayers-1)
	biases := make([]*mat.Dense, numLayers-1)
	//Loops through the number of layers
	for i := 1; i < numLayers; i++ {
		//Defines the Matrices
		weight := mat.NewDense(sizes[i], sizes[i-1], nil)
		bias := mat.NewDense(sizes[i], 1, nil)

		//Sets random values for the biases and weights
		for i := 0; i < bias.RawMatrix().Rows; i++ {
			val := rand.Float64()
			bias.Set(i, 0, val)
		}

		for row := 0; row < sizes[i]; row++ {
			for col := 0; col < sizes[i-1]; col++ {
				val := rand.Float64()
				weight.Set(row, col, val)
			}
		}

		weights[i-1] = weight
		biases[i-1] = bias
	}
	//returns a network
	return &network{numLayers: numLayers, sizes: sizes, weights: weights, biases: biases}
}

// Sigmoid function
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(z))
}

// Derivative of sigmoid functions
func sigmoidPrime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

// Cost function derivative
func (nn *network) cost_derivative(output_a *mat.Dense, y mnist.Label) *mat.Dense {
	e := mat.NewDense(10, 1, nil)
	e.Set(int(y), 0, 1.0)
	output_a.Sub(output_a, e)

	return output_a
}

// FeedForward Function - feeds forward through the network and returns an output
func (nn *network) feedforward(a *mat.Dense) *mat.Dense {
	//Loops through the network (specifically)
	for i := 0; i < len(nn.weights); i++ {
		//Implements the equation a' = sigmoid(w*a + b)
		var z mat.Dense
		weights := nn.weights[i]
		biases := nn.biases[i]
		z.Mul(weights, a)
		z.Add(&z, biases)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}

		a = &z
		a.Apply(applySigmoid, &z)

	}
	return a
}

// Used for Shuffling the training data to give a random training set
func ShuffleTrainingData(training_data *mnist.Set) {
	n := len(training_data.Images)

	perm := rand.Perm(n)

	shuffledImages := make([]*mnist.Image, n)
	shuffledLabels := make([]mnist.Label, n)

	for i, idx := range perm {
		shuffledImages[i] = training_data.Images[idx]
		shuffledLabels[i] = training_data.Labels[idx]
	}

	training_data.Images = shuffledImages
	training_data.Labels = shuffledLabels
}

// Starts the process of learning usign stochasitc gradient descent. Intakes the size of a mini-batch of training data
// the number of epochs of training, the learning rate eta, and then the training data and test data
func (nn *network) StochasticGradientDescent(mini_batch_size int, epochs int, eta float64, training_data *mnist.Set, test_data *mnist.Set) {
	n_test := test_data.Count()
	n := training_data.Count()
	//Loop through the epochs
	for i := 0; i < epochs; i++ {
		ShuffleTrainingData(training_data)
		//Creates a list of mini batches for storing a mini batch
		mini_batches := make([]*mnist.Set, 0)
		//Populates the list of mini batches
		for k := 0; k < n; k += mini_batch_size {
			end := k + mini_batch_size
			if end > n {
				end = n
			}
			mini_batch := mnist.Set{
				Images: training_data.Images[k:end],
				Labels: training_data.Labels[k:end],
			}
			mini_batches = append(mini_batches, &mini_batch)
		}
		//Starts the process of learning using a mini batch to update the weights and biases
		for _, mini_batch := range mini_batches {
			nn.update_mini_batch(mini_batch, eta)
		}
		fmt.Printf("Epoch[%d]\n", i)
		res := nn.evaluate(test_data)
		fmt.Printf("Accuracy: %d / %d\n", res, n_test)
	}
}

// Continues the process of learning by intaking a mini_batch and the learning rate and updates the network
func (nn *network) update_mini_batch(mini_batch *mnist.Set, eta float64) {
	//Creates a nabla biases list of matrices which contains a list of matrices equal to length and size of the biases
	//all set to 0
	nabla_b := make([]*mat.Dense, len(nn.biases))
	for i := range nn.biases {
		nabla_b[i] = mat.NewDense(nn.biases[i].RawMatrix().Rows, 1, nil)

	}
	//Same with nabla w
	nabla_w := make([]*mat.Dense, len(nn.weights))
	for i := range nn.weights {
		nabla_w[i] = mat.NewDense(nn.weights[i].RawMatrix().Rows, nn.weights[i].RawMatrix().Cols, nil)

	}
	//Defines delta_nabla_b and delta_nabla_w using the back propagation algorithm
	for i := 0; i < mini_batch.Count(); i++ {
		delta_nabla_b, delta_nabla_w := nn.backprop(mini_batch.Images[i], mini_batch.Labels[i])
		for j := range nabla_b {
			//Starts to update the weights and biases by adding the nablas with the delta nablas
			nabla_b[j].Add(nabla_b[j], delta_nabla_b[j])
			nabla_w[j].Add(nabla_w[j], delta_nabla_w[j])
			// fmt.Println(mat.Formatted(nabla_b[j]))
			// time.Sleep(10000000)
		}
	}
	// fmt.Println("New batch")
	// fmt.Println(eta / float64(mini_batch.Count()) * nabla_w[1].At(5, 2))
	//Sets the weights and biases to the updated values
	for i := 0; i < len(nn.weights); i++ {
		for j := 0; j < nn.weights[i].RawMatrix().Rows; j++ {
			for k := 0; k < nn.weights[i].RawMatrix().Cols; k++ {
				val := nn.weights[i].At(j, k) - (eta/float64(mini_batch.Count()))*nabla_w[i].At(j, k)
				nn.weights[i].Set(j, k, val)

			}
		}
		// fmt.Println("Weights: ")
		// fmt.Println(nn.weights[1].At(5, 2))
		// time.Sleep(100000000)
	}

	for i := 0; i < len(nn.biases); i++ {
		for j := 0; j < nn.biases[i].RawMatrix().Rows; j++ {
			for k := 0; k < nn.biases[i].RawMatrix().Cols; k++ {
				val := nn.biases[i].At(j, k) - (eta/float64(mini_batch.Count()))*nabla_b[i].At(j, k)
				nn.biases[i].Set(j, k, val)
			}
		}
	}
}

// The backpropagation algorithm intakes the input and label and feeds forward the input
// and runs back through the network adjusting the weights and biases
func (nn *network) backprop(x *mnist.Image, y mnist.Label) ([]*mat.Dense, []*mat.Dense) {
	//Defines nabla_b adn nabla_w and populates them with 0s
	nabla_b := make([]*mat.Dense, len(nn.biases))
	for i := range nn.biases {
		nabla_b[i] = mat.NewDense(nn.biases[i].RawMatrix().Rows, 1, nil)

	}
	nabla_w := make([]*mat.Dense, len(nn.weights))
	for i := range nn.weights {
		nabla_w[i] = mat.NewDense(nn.weights[i].RawMatrix().Rows, nn.weights[i].RawMatrix().Cols, nil)

	}
	//Creates a usuable input matrix with each row being a pixel with a a value at column 0 pretaining to the color value from 0 - 255
	a := mat.NewDense(len(x), 1, nil)
	for i := 0; i < len(x); i++ {
		val := float64(x[i])
		a.Set(i, 0, val/255)
	}
	zs := make([]*mat.Dense, nn.numLayers-1)
	activations := make([]*mat.Dense, nn.numLayers)
	activations[0] = a
	for i := 0; i < len(nn.weights); i++ {
		var z mat.Dense
		weights := nn.weights[i]
		biases := nn.biases[i]
		z.Mul(weights, a)
		z.Add(&z, biases)
		zs[i] = &z
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}

		z.Apply(applySigmoid, &z)
		a = &z
		activations[i+1] = a
	}

	delta := nn.cost_derivative(activations[len(activations)-1], y)

	for i := 0; i < delta.RawMatrix().Rows; i++ {
		for j := 0; j < delta.RawMatrix().Cols; j++ {
			delta.Set(i, j, delta.At(i, j)*sigmoidPrime(zs[len(zs)-1].At(i, j)))
		}
	}

	nabla_b[len(nabla_b)-1] = delta
	nabla_w[len(nabla_w)-1].Mul(delta, activations[len(activations)-1].T())

	for l := 2; l < nn.numLayers; l++ {
		z := zs[len(zs)-l]
		fmt.Println("__________________________")
		fmt.Println(mat.Formatted(delta))
		temp := z
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return sigmoidPrime(v)
		}
		temp.Apply(applySigmoidPrime, z)
		fmt.Println(mat.Formatted(delta))
		delta.Mul(nn.weights[len(nn.weights)-l+1].T(), delta)
		fmt.Println(mat.Formatted(delta))
		delta.MulElem(delta, temp)
		fmt.Println(mat.Formatted(delta))
		nabla_b[len(nabla_b)-l] = delta
		nabla_w[len(nabla_w)-l].Mul(delta, activations[len(activations)-l-1].T())
	}
	return nabla_b, nabla_w
}

func (nn *network) evaluate(test_data *mnist.Set) int {
	test_results := make([]float64, test_data.Count())
	for i := 0; i < test_data.Count(); i++ {
		e := mat.NewDense(10, 1, nil)
		e.Set(int(*&test_data.Labels[i]), 0, 1.0)
		a := mat.NewDense(784, 1, nil)
		for j := 0; j < 784; j++ {
			val := float64(test_data.Images[i][j])
			a.Set(j, 0, val/255)
		}
		test_result := nn.feedforward(a)
		maxVal := test_result.At(0, 0)
		num := 0.0
		for i := 0; i < test_result.RawMatrix().Rows; i++ {
			val := test_result.At(i, 0)
			if val > maxVal {
				maxVal = val
				num = float64(i)
			}
		}
		test_results[i] = num
	}
	sum := 0
	for i := 0; i < len(test_results); i++ {
		if test_results[i] == float64(test_data.Labels[i]) {
			sum++
		}
	}

	return sum
}

func main() {
	fmt.Println()
	sizes := []int{784, 10, 10}
	nn := newNetwork(3, sizes)

	training_data, test_data, err := mnist.Load("./data/")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Starting")
	nn.StochasticGradientDescent(30, 10, 3.0, training_data, test_data)

}

//TODO:
//Change the input from the data set from 0 - 255 grayscale to 0.0 - 1.0 by dividing each pixels color value by 255
//This will involve changing a couple of things
