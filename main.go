package main

import (
	"flag"
	"fmt"
)

func main() {
	var loop int
	var file string
	var mode string
	var model string
	flag.IntVar(&loop, "l", 10, "number of iterations")
	flag.StringVar(&file, "f", "", "data file")
	flag.StringVar(&mode, "m", "", "mode {learn, test}")
	flag.StringVar(&model, "w", "", "model file")
	flag.Parse()

	if file == "" {
		panic("Data must be specified")
	}

	if mode == "learn" {
		flag.Parse()
		X, y := LoadFromFile(file)
		p := Perceptron{map[string]map[string]float64{}, loop}
		p.Fit(X, y)
		SaveModel(p.weight, model)
	} else if mode == "test" {
		p := Perceptron{map[string]map[string]float64{}, loop}
		weight := LoadModel(model)
		p.weight = weight

		X_test, y_test := LoadFromFile(file)
		num_corr := 0.
		n := 0.
		for i, X_i := range X_test {
			pred_y_i := p.Predict(X_i)
			if pred_y_i == y_test[i] {
				num_corr += 1
			}
			n += 1
		}
		acc := num_corr / n
		fmt.Println("Acc:", acc)
	} else {
		panic("Invalid mode")
	}
}
