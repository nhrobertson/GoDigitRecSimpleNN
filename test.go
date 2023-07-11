// package main

// import (
// 	"fmt"
// 	"math"

// 	"gonum.org/v1/gonum/mat"
// )

// type NeuralNetwork struct {
// 	biases  []*mat.Dense
// 	weights []*mat.Dense
// }

// func sigmoid(x float64) float64 {
// 	return 1.0 / (1.0 + math.Exp(-x))
// }

// func (nn *NeuralNetwork) feedforward(a *mat.Dense) *mat.Dense {
// 	// Iterate over biases and weights simultaneously
// 	for i := 0; i < len(nn.biases); i++ {
// 		b := nn.biases[i]
// 		w := nn.weights[i]

// 		// Perform matrix multiplication and addition
// 		var z mat.Dense
// 		z.Mul(w, a)
// 		z.Add(&z, b)

// 		// Apply sigmoid function element-wise
// 		applySigmoid := func(_, _ int, v float64) float64 {
// 			return sigmoid(v)
// 		}
// 		a.Apply(applySigmoid, &z)
// 	}

// 	return a
// }

// func main() {
// 	// Example usage
// 	nn := NeuralNetwork{
// 		biases: []*mat.Dense{
// 			mat.NewDense(3, 1, []float64{0.5, -0.5, 0.1}),
// 			mat.NewDense(2, 1, []float64{-0.2, 0.3}),
// 		},
// 		weights: []*mat.Dense{
// 			mat.NewDense(3, 2, []float64{0.1, 0.2, 0.3, -0.4, 0.5, 0.6}),
// 			mat.NewDense(2, 3, []float64{-0.7, 0.8, 0.9, 1.0, -1.1, -1.2}),
// 		},
// 	}

// 	input := mat.NewDense(2, 1, []float64{0.2, 0.4})
// 	output := nn.feedforward(input)

// 	fmt.Println("Output:")
// 	fmt.Println(mat.Formatted(output))
// }
