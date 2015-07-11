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
		//         eta       float64
		//         c         float64
		loop     int
		testfile string
	)
	//     fmt.Println(args)

	fs := flag.NewFlagSet("train", flag.ExitOnError)
	fs.StringVar(&model, "model", "", "model filename")
	fs.StringVar(&model, "m", "", "model filename")
	fs.StringVar(&testfile, "t", "", "test file")
	fs.StringVar(&algorithm, "algorithm", "", "algorithm for training {perceptron, pa2, adagrad")
	fs.StringVar(&algorithm, "a", "", "algorithm for training {perceptron, pa, pa1, pa2, adagrad")
	//     fs.Float64Var(&eta, "eta", 0.1, "learning rate")
	//     fs.Float64Var(&c, "c", 1e-5, "regularization parameter")
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

	switch algorithm {
	case "perceptron":
		learner = gonline.NewPerceptron()
	case "pa":
		learner = gonline.NewPA("")
	case "pa1":
		learner = gonline.NewPA("I")
	case "pa2":
		learner = gonline.NewPA("II")
	case "cw":
		learner = gonline.NewCW()
	default:
		panic("Invalid algorithm")
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
