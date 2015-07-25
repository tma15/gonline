package main

import (
	"flag"
	"fmt"
	"github.com/tma15/gonline"
	"os"
	"runtime/pprof"
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

	var (
		learner gonline.LearnerInterface
		x_train *[]map[string]float64
		y_train *[]string
		x_test  *[]map[string]float64
		y_test  *[]string
	)

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
		x_test, y_test = gonline.LoadData(testfile)
	}

	for i := 0; i < loop; i++ {
		for _, trainfile := range fs.Args() {
			x_train, y_train = gonline.LoadData(trainfile)
			gonline.ShuffleData(x_train, y_train)
			learner.Fit(x_train, y_train)
			if testfile != "" {
				numCorr := 0
				numTotal := 0
				cls := gonline.Classifier{}
				cls.Weight = *learner.GetParam()
				ftdic, labeldic := learner.GetDics()
				cls.FtDict = *ftdic
				cls.LabelDict = *labeldic
				for i, x_i := range *x_test {
					j := cls.Predict(&x_i)
					if cls.LabelDict.Id2elem[j] == (*y_test)[i] {
						numCorr += 1
					}
					numTotal += 1
				}
				acc := float64(numCorr) / float64(numTotal)
				fmt.Printf("epoch:%d test accuracy: %f (%d/%d)\n", i+1, acc, numCorr, numTotal)
			}
		}
	}
	learner.Save(model)
}

func test(args []string) {
	var model string
	fs := flag.NewFlagSet("test", flag.ExitOnError)
	fs.StringVar(&model, "model", "", "model filename")
	fs.StringVar(&model, "m", "", "model filename")
	fs.Parse(args)

	cls := gonline.LoadClassifier(model)

	numCorr := 0
	numTotal := 0
	for _, fname := range fs.Args() {
		x, y := gonline.LoadData(fname)
		for i, x_i := range *x {
			j := cls.Predict(&x_i)
			if cls.LabelDict.Id2elem[j] == (*y)[i] {
				numCorr += 1
			}
			numTotal += 1
		}
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

	prof := "mycpu.prof"
	f, err := os.Create(prof)
	if err != nil {
		panic(err)
	}
	err = pprof.StartCPUProfile(f)
	if err != nil {
		panic(err)
	}
	defer pprof.StopCPUProfile()

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
