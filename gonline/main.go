package main

import (
	"flag"
	"fmt"
	"github.com/tma15/gonline"
	"os"
	"runtime"
	"runtime/pprof"
	//         "sync"
)

func ChooseLearner(algorithm string, param map[string]float64) gonline.LearnerInterface {
	var learner gonline.LearnerInterface
	C := param["C"]
	eta := param["eta"]
	gamma := param["gamma"]
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
	case "adam":
		learner = gonline.NewAdam()
	default:
		panic(fmt.Sprintf("Invalid algorithm: %s", algorithm))
	}
	return learner
}

func EvalLearner(epoch int, learner *gonline.LearnerInterface, x_test *[]map[string]float64, y_test *[]string) {
	numCorr := 0
	numTotal := 0
	cls := gonline.Classifier{}
	w := (*learner).GetParam()
	ft, label := (*learner).GetDics()
	cls.Weight = *w
	cls.FtDict = *ft
	cls.LabelDict = *label
	for i, x_i := range *x_test {
		j := cls.Predict(&x_i)
		if cls.LabelDict.Id2elem[j] == (*y_test)[i] {
			numCorr += 1
		}
		numTotal += 1
	}
	acc := float64(numCorr) / float64(numTotal)
	fmt.Printf("epoch:%d test accuracy: %f (%d/%d)\n", epoch, acc, numCorr, numTotal)
}

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

	param := make(map[string]float64)
	param["C"] = C
	param["eta"] = eta
	param["gamma"] = gamma

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

	var learner_avg *gonline.LearnerInterface
	switch trainStrategy {
	case "ipm":
		runtime.GOMAXPROCS(numCpu)
		fmt.Println("number of cpu cores", numCpu)
		learners := make([]gonline.LearnerInterface, numCpu, numCpu)
		for i := 0; i < numCpu; i++ {
			learners[i] = ChooseLearner(algorithm, param)
		}
		fmt.Println("algorithm:", learners[0].Name())
		runtime.GOMAXPROCS(numCpu)
		for _, trainfile := range fs.Args() {
			x_train, y_train = gonline.LoadData(trainfile)
			if !without_shuffle {
				gonline.ShuffleData(x_train, y_train)
			}
			for i := 0; i < loop; i++ {
				if numCpu > 1 {
					gonline.FitLearners(&learners, x_train, y_train)
					learner_avg = gonline.AverageModels(learners)
					gonline.BroadCastModel(learner_avg, &learners)
				} else {
					learners[0].Fit(x_train, y_train)
					learner_avg = &learners[0]
				}
				if testfile != "" {
					EvalLearner(i+1, learner_avg, x_test, y_test)
				}
			}
		}
		(*learner_avg).Save(model)
	default:
		learner = ChooseLearner(algorithm, param)
		fmt.Println("algoriths:", learner.Name())
		for i := 0; i < loop; i++ {
			for _, trainfile := range fs.Args() {
				x_train, y_train = gonline.LoadData(trainfile)
				if !without_shuffle {
					gonline.ShuffleData(x_train, y_train)
				}
				learner.Fit(x_train, y_train)
			}
			if testfile != "" {
				EvalLearner(i+1, &learner, x_test, y_test)
			}
		}
		//                 learner.Save(model)
		learner.SaveBinary(model)
	}
}

func test(args []string) {
	var model string
	var verbose bool
	fs := flag.NewFlagSet("test", flag.ExitOnError)
	fs.StringVar(&model, "model", "", "model filename")
	fs.StringVar(&model, "m", "", "model filename")
	fs.BoolVar(&verbose, "v", false, "verbose mode")
	fs.Parse(args)

	//         cls := gonline.LoadClassifier(model)
	cls := gonline.LoadClassifierBinary(model)

	numCorr := 0
	numTotal := 0
	for _, fname := range fs.Args() {
		x, y := gonline.LoadData(fname)
		for i, x_i := range *x {
			j := cls.Predict(&x_i)
			if cls.LabelDict.Id2elem[j] == (*y)[i] {
				numCorr += 1
			}
			if verbose {
				fmt.Println(cls.LabelDict.Id2elem[j], (*y)[i])
			}
			numTotal += 1
		}
	}
	acc := float64(numCorr) / float64(numTotal)
	fmt.Printf("test accuracy: %f (%d/%d)\n", acc, numCorr, numTotal)

}

// func DistributedFitting() {
//         hosts := []string{"127.0.0.1"}
//         ports := []string{"8888"}
//     hosts := []string{"127.0.0.1", "127.0.0.1"}
//     ports := []string{"8888", "8889"}
//     hosts := []string{"127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"}
//     ports := []string{"8888", "8889", "8890", "8891"}

//         numServers := len(hosts)
//         for i := 0; i < numServers; i++ {
//                 s := gonline.NewLearnerServer(hosts[i], ports[i])
//                 go s.Start()
//         }

//         x, y := gonline.LoadData("./news20.scale")
//         data := gonline.Data{
//                 X: x,
//                 Y: y,
//         }

//         num_data := len(*data.X)
//         sizechunk := num_data/numServers + 1

//         cli := gonline.NewClient()

//         var avg_learner *gonline.LearnerInterface
//         numLoop := 1
//         for t := 0; t < numLoop; t++ {
//                 learners := make(chan *gonline.LearnerInterface)
//                 buffer := make(chan int)
//                 var wg sync.WaitGroup
//                 wg.Add(1)
//                 go func(procs chan int) {
//                         defer wg.Done()
//                         for pid := range procs {
//                                 start := pid * sizechunk
//                                 end := (pid + 1) * sizechunk
//                                 if end >= num_data {
//                                         end = num_data - 1
//                                 }
//                                 data_batch := data.GetBatch(start, end)

//                                 learner := cli.SendData(hosts[pid], ports[pid], data_batch)
//                                 learners <- learner
//                         }
//                 }(buffer)

//                 go func() {
//                         for i := 0; i < numServers; i++ {
//                                 buffer <- i
//                         }
//                         close(buffer)
//                         wg.Wait()
//                         close(learners)
//                 }()

//                 _learners := make([]gonline.LearnerInterface, numServers, numServers)
//                 i := 0
//                 for learner := range learners {
//                         _learners[i] = *learner
//                         i++
//                 }
//                 avg_learner = gonline.AverageModels(_learners)

//                 buffer2 := make(chan int)
//                 var wg2 sync.WaitGroup
//                 wg2.Add(1)
//                 go func(procs chan int) {
//                         defer wg2.Done()
//                         for pid := range procs {
//                                 cli.SendModel(hosts[pid], ports[pid], avg_learner)
//                         }
//                 }(buffer2)
//                 go func() {
//                         for i := 0; i < numServers; i++ {
//                                 buffer2 <- i
//                         }
//                         close(buffer2)
//                         wg2.Wait()
//                 }()
//         }

//         (*avg_learner).Save("model")

//         cls := gonline.LoadClassifier("model")
//         numCorr := 0
//         numTotal := 0
//         x, y = gonline.LoadData("./news20.t.scale")
//         for i, x_i := range *x {
//                 j := cls.Predict(&x_i)
//                 if cls.LabelDict.Id2elem[j] == (*y)[i] {
//                         numCorr += 1
//                 }
//                 numTotal += 1
//         }
//         acc := float64(numCorr) / float64(numTotal)
//         fmt.Printf("test accuracy: %f (%d/%d)\n", acc, numCorr, numTotal)
// }

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
		//         case "dist":
		//                 DistributedFitting()
	default:
		flag.Usage()
		os.Exit(1)
	}
}
