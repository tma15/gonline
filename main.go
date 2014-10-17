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
	var eta float64
	var verbose bool
	flag.IntVar(&loop, "l", 10, "number of iterations")
	flag.Float64Var(&eta, "a", 0.5, "learning rate")
	flag.StringVar(&file, "f", "", "data file")
	flag.StringVar(&mode, "m", "", "mode {learn, test}")
	flag.StringVar(&model, "w", "", "model file")
	flag.BoolVar(&verbose, "v", false, "verbose mode")
	flag.Parse()

	if file == "" {
		panic("Data must be specified")
	}

	if mode == "learn" {
		flag.Parse()
		X, y := LoadFromFile(file)
		p := NewPerceptron(eta, loop)
		p.Fit(X, y)
		SaveModel(p, model)
	} else if mode == "test" {
                p := LoadModel(model)

		X_test, y_test := LoadFromFile(file)
		num_corr := 0.
		n := 0.
                pred_y := []string{}
		for i, X_i := range X_test {
			pred_y_i := p.Predict(X_i)
                        pred_y = append(pred_y, pred_y_i)
			if verbose {
				fmt.Println(pred_y_i)
			}
			if pred_y_i == y_test[i] {
				num_corr += 1
			}
			n += 1
		}
		acc := num_corr / n
                confusionMatrix(y_test, pred_y)
		fmt.Println("Acc:", acc)
	} else {
		panic("Invalid mode")
	}
}
