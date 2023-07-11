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

	weights := make([]*mat.Dense, numLayers-1)
	biases := make([]*mat.Dense, numLayers-1)
	for i := 1; i < numLayers; i++ {
		weight := mat.NewDense(sizes[i], sizes[i-1], nil)
		bias := mat.NewDense(sizes[i], 1, nil)
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

func (nn *network) feedforward(a *mat.Dense) *mat.Dense {
	for i := 0; i < len(nn.weights); i++ {
		var z mat.Dense
		weights := nn.weights[i]
		biases := nn.biases[i]
		z.Mul(weights, a)
		z.Add(&z, biases)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}
		fmt.Printf("Z Matrix [%d]\n", i)
		fmt.Println(mat.Formatted(&z))
		a = &z
		a.Apply(applySigmoid, &z)
		fmt.Printf("A Matrix [%d]\n", i)
		fmt.Println(mat.Formatted(a))
	}
	return a
}

func (nn *network) StochasticGradientDescent(mini_batch_size int, epochs int, eta float64, training_data *mnist.Set, test_data *mnist.Set) {
	if test_data.Count() != 0 {
		//n_test := test_data.Count()
	}
	n := training_data.Count()
	for i := 0; i < epochs; i++ {
		mini_batches := make([]*mnist.Set, 0)
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
		for _, mini_batch := range mini_batches {
			nn.update_mini_batch(mini_batch, eta)
		}
		fmt.Printf("Epoch[%d]", i)
		fmt.Println(mini_batches[i])

	}
}

func (nn *network) update_mini_batch(mini_batch *mnist.Set, eta float64) {
	nabla_b := make([]*mat.Dense, len(nn.biases))
	for i := range nn.biases {
		nabla_b[i] = mat.NewDense(nn.biases[i].RawMatrix().Rows, 1, nil)
		// fmt.Printf("nabla_b[%d]:\n", i)
		// fmt.Println(mat.Formatted(nabla_b[i]))
	}
	nabla_w := make([]*mat.Dense, len(nn.weights))
	for i := range nn.weights {
		nabla_w[i] = mat.NewDense(nn.weights[i].RawMatrix().Rows, nn.weights[i].RawMatrix().Cols, nil)
		// fmt.Printf("nabla_w[%d]:\n", i)
		// fmt.Println(mat.Formatted(nabla_w[i]))
	}
	for i := 0; i < mini_batch.Count(); i++ {
		delta_nabla_b, delta_nabla_w := nn.backprop(mini_batch.Images[i], mini_batch.Labels[i])
		for j := range nabla_b {
			nabla_b[j].Add(nabla_b[j], delta_nabla_b[j])
			nabla_w[j].Add(nabla_w[j], delta_nabla_w[j])
		}
	}
	for i := 0; i < len(nn.weights); i++ {
		for j := 0; j < nn.weights[i].RawMatrix().Rows; j++ {
			for k := 0; k < nn.weights[i].RawMatrix().Cols; k++ {
				val := nn.weights[i].At(j, k) - (eta/float64(mini_batch.Count()))*nabla_w[i].At(j, k)
				nn.weights[i].Set(j, k, val)
			}
		}
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

func (nn *network) backprop(x *mnist.Image, y mnist.Label) ([]*mat.Dense, []*mat.Dense) {
	nabla_b := make([]*mat.Dense, len(nn.biases))
	for i := range nn.biases {
		nabla_b[i] = mat.NewDense(nn.biases[i].RawMatrix().Rows, 1, nil)
		// fmt.Printf("nabla_b[%d]:\n", i)
		// fmt.Println(mat.Formatted(nabla_b[i]))
	}
	nabla_w := make([]*mat.Dense, len(nn.weights))
	for i := range nn.weights {
		nabla_w[i] = mat.NewDense(nn.weights[i].RawMatrix().Rows, nn.weights[i].RawMatrix().Cols, nil)
		// fmt.Printf("nabla_w[%d]:\n", i)
		// fmt.Println(mat.Formatted(nabla_w[i]))
	}

	//TODO:
	return nabla_b, nabla_w
}

func main() {
	fmt.Println()
	sizes := []int{784, 10, 10}
	nn := newNetwork(3, sizes)
	fmt.Println("Weights Matrix[0]")
	fmt.Println(mat.Formatted(nn.weights[0]))
	fmt.Println("Weights Matrix [1]")
	fmt.Println(mat.Formatted(nn.weights[1]))
	fmt.Println("Biase Vector[0]")
	fmt.Println(mat.Formatted(nn.biases[0]))
	fmt.Println("Bias Vector[1]")
	fmt.Println(mat.Formatted(nn.biases[1]))
	fmt.Printf("\nNeural Network: %d, Sizes: %d, %d, %d\n\n", nn.numLayers, nn.sizes[0], nn.sizes[1], nn.sizes[2])
	fmt.Println(" ")

	a := mat.NewDense(sizes[0], 1, nil)
	for i := 0; i < a.RawMatrix().Rows; i++ {
		for j := 0; j < a.RawMatrix().Cols; j++ {
			a.Set(i, j, rand.Float64())
		}
	}
	fmt.Println("Original A Matrix: ")
	fmt.Println(mat.Formatted(a))

	output := nn.feedforward(a)
	fmt.Print("\nOutput: \n")
	fmt.Println(mat.Formatted(output))

	training_data, test_data, err := mnist.Load("./data/")
	if err != nil {
		log.Fatal(err)
	}

	nn.StochasticGradientDescent(10, 10, 0.1, training_data, test_data)
	// fmt.Println(training_data)
	// fmt.Println(test_data)

}
