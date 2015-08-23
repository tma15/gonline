package main

import (
	"flag"
	"fmt"
	"github.com/tma15/gonline"
	"os"
	"runtime"
	"runtime/pprof"
)

func train(args []string) {
	var (
		model           string
		algorithm       string
		eta             float64
		gamma           float64
		C               float64
		loop            int
		numCpu          int
		testfile        string
		trainStrategy   string
		without_shuffle bool
	)

	fs := flag.NewFlagSet("train", flag.ExitOnError)
	fs.StringVar(&model, "model", "", "file name of model")
	fs.StringVar(&model, "m", "", "file name of model")
	fs.StringVar(&testfile, "t", "", "file name of test data")
	fs.StringVar(&algorithm, "algorithm", "", "algorithm for training {p, pa, pa1, pa2, cw, arow}")
	fs.StringVar(&algorithm, "a", "", "algorithm for training {p, pa, pa1, pa2, cw, arow}")
	fs.StringVar(&trainStrategy, "s", "", "training strategy {ipm}; default is training with single core")
	fs.Float64Var(&eta, "eta", 0.8, "confidence parameter for Confidence Weighted")
	fs.Float64Var(&gamma, "g", 10., "regularization parameter for AROW")
	fs.Float64Var(&C, "C", 0.01, "degree of aggressiveness for PA-I and PA-II")
	fs.IntVar(&loop, "i", 1, "number of iterations")
	fs.IntVar(&numCpu, "p", runtime.NumCPU(), "number of cores for ipm (Iterative Prameter Mixture)")
	fs.BoolVar(&without_shuffle, "withoutshuffle", false, "doesn't shuffle the training data")
	fs.Parse(args)

	var (
		learner gonline.LearnerInterface
		x_train *[]map[string]float64
		y_train *[]string
		x_test  *[]map[string]float64
		y_test  *[]string
	)

	if testfile != "" {
		fmt.Println("testfile", testfile)
		x_test, y_test = gonline.LoadData(testfile)
	}
	if without_shuffle {
		fmt.Println("training data will not be shuffled")
	} else {
		fmt.Println("training data will be shuffled")
	}
	if trainStrategy != "" {
		fmt.Println("training strategy:", trainStrategy)
	}

	switch trainStrategy {
	case "ipm":
		runtime.GOMAXPROCS(numCpu)
		fmt.Println("number of cpu cores", numCpu)
		learners := make([]gonline.LearnerInterface, numCpu, numCpu)
		for i := 0; i < numCpu; i++ {
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
			learners[i] = learner
		}
		fmt.Println("algorithm:", learners[0].Name())
		runtime.GOMAXPROCS(numCpu)
		for i := 0; i < loop; i++ {
			for _, trainfile := range fs.Args() {
				x_train, y_train = gonline.LoadData(trainfile)
				if !without_shuffle {
					gonline.ShuffleData(x_train, y_train)
				}
				gonline.FitLearners(&learners, x_train, y_train)
				learner_avg := gonline.AverageModels(&learners)
				gonline.BroadCastModel(learner_avg, &learners)
				if testfile != "" {
					numCorr := 0
					numTotal := 0
					cls := gonline.Classifier{}
					avg_w := learners[0].GetParam()
					avg_ft, avg_label := learners[0].GetDics()
					cls.Weight = *avg_w
					cls.FtDict = *avg_ft
					cls.LabelDict = *avg_label
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
	default:
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
		fmt.Println("algorithm:", learner.Name())
		for i := 0; i < loop; i++ {
			for _, trainfile := range fs.Args() {
				x_train, y_train = gonline.LoadData(trainfile)
				if !without_shuffle {
					gonline.ShuffleData(x_train, y_train)
				}
				learner.Fit(x_train, y_train)

			}
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
