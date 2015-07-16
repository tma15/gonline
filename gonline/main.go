package main

import (
	"flag"
	"fmt"
	"github.com/tma15/gonline"
	"os"
)

func train(args []string) {
	var (
		model     string
		algorithm string
		eta       float64
		gamma     float64
		C         float64
		loop      int
		testfile  string
	)

	fs := flag.NewFlagSet("train", flag.ExitOnError)
	fs.StringVar(&model, "model", "", "model filename")
	fs.StringVar(&model, "m", "", "model filename")
	fs.StringVar(&testfile, "t", "", "test file")
	fs.StringVar(&algorithm, "algorithm", "", "algorithm for training {p, pa, pa1, pa2, cw, arow}")
	fs.StringVar(&algorithm, "a", "", "algorithm for training {p, pa, pa1, pa2, cw, arow}")
	fs.Float64Var(&eta, "eta", 0.8, "confidence parameter for Confidence Weighted")
	fs.Float64Var(&gamma, "g", 10., "regularization parameter for AROW")
	fs.Float64Var(&C, "C", 0.01, "degree of aggressiveness for PA-I and PA-II")
	fs.IntVar(&loop, "i", 1, "iteration number")
	fs.Parse(args)

	//     fmt.Println(fs.Args())
	var (
		learner   gonline.LearnerInterface
		ftdict    gonline.Dict
		labeldict gonline.Dict
		x_data    [][]gonline.Feature
		y_data    []int
	)
	ftdict = gonline.NewDict()
	labeldict = gonline.NewDict()

	fmt.Println("algorithm:", algorithm)
	switch algorithm {
	case "p":
		learner = gonline.NewPerceptron()
	case "pa":
		learner = gonline.NewPA("", C)
	case "pa1":
		learner = gonline.NewPA("I", C)
	case "pa2":
		learner = gonline.NewPA("II", C)
	case "cw":
		learner = gonline.NewCW(eta)
	case "arow":
		learner = gonline.NewArow(gamma)
	default:
		panic(fmt.Sprintf("Invalid algorithm: %s", algorithm))
	}

	if testfile != "" {
		fmt.Println("testfile", testfile)
	}

	for i := 0; i < loop; i++ {
		for _, trainfile := range fs.Args() {
			gonline.LoadTrainData(trainfile, &ftdict, &labeldict, &x_data, &y_data)
			gonline.ShuffleData(&x_data, &y_data)
			learner.Fit(&x_data, &y_data)
			if testfile != "" {
				x_test := make([][]gonline.Feature, 0, 10000)
				y_test := make([]int, 0, 10000)
				gonline.LoadTestData(testfile, &ftdict, &labeldict, &x_test, &y_test)
				numCorr := 0
				numTotal := 0
				cls := gonline.Classifier{}
				cls.Weight = *learner.GetParam()
				for i, x_i := range x_test {
					j := cls.Predict(&x_i)
					if j == y_test[i] {
						numCorr += 1
					}
					numTotal += 1
				}
				acc := float64(numCorr) / float64(numTotal)
				fmt.Printf("epoch:%d test accuracy: %f (%d/%d)\n", i+1, acc, numCorr, numTotal)
			}
		}
	}
	learner.Save(model, &ftdict, &labeldict)

	/* closed test */
	cls := gonline.Classifier{}
	cls.Weight = *learner.GetParam()
	numCorr := 0
	numTotal := 0
	x := make([][]gonline.Feature, 0, 100000)
	y := make([]int, 0, 100000)
	for _, trainfile := range fs.Args() {
		gonline.LoadTestData(trainfile, &ftdict, &labeldict, &x, &y)
		for i, x_i := range x {
			j := cls.Predict(&x_i)
			//             fmt.Println(j, y[i])
			if j == y[i] {
				numCorr += 1
			}
			numTotal += 1
		}
	}
	acc := float64(numCorr) / float64(numTotal)
	fmt.Printf("train accuracy: %f (%d/%d)\n", acc, numCorr, numTotal)

}

func test(args []string) {
	var model string
	fs := flag.NewFlagSet("test", flag.ExitOnError)
	fs.StringVar(&model, "model", "", "model filename")
	fs.StringVar(&model, "m", "", "model filename")
	fs.Parse(args)

	ftdict := gonline.NewDict()
	labeldict := gonline.NewDict()
	cls := gonline.LoadClassifier(model, &ftdict, &labeldict)

	numCorr := 0
	numTotal := 0
	x := make([][]gonline.Feature, 0, 100000)
	y := make([]int, 0, 100000)
	for _, fname := range fs.Args() {
		gonline.LoadTestData(fname, &ftdict, &labeldict, &x, &y)
		for i, x_i := range x {
			j := cls.Predict(&x_i)
			if j == y[i] {
				numCorr += 1
			}
			numTotal += 1
		}
		//         gonline.ConfusionMatrix(y, predy)
	}
	acc := float64(numCorr) / float64(numTotal)
	fmt.Printf("test accuracy: %f (%d/%d)\n", acc, numCorr, numTotal)

}

var usage = `
Usage of %s <Command> [Options]

Commands:
  train
  test

`

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, usage, os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()
	args := flag.Args()

	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	switch args[0] {
	case "train":
		train(args[1:])
	case "test":
		test(args[1:])
	default:
		flag.Usage()
		os.Exit(1)
	}
}
