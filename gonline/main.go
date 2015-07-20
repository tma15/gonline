package main

import (
	"flag"
	"fmt"
	"github.com/tma15/gonline"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func update_parallel(i, avg_num int, learner *gonline.LearnerInterface, x *[][]gonline.Feature, y *[]int, x_p *[][]gonline.Feature, y_p *[]int) *[][]float64 {
	offset_start := avg_num * i
	offset_end := offset_start + avg_num - 1
	k := 0
	start := time.Now()
	for i := offset_start; i < offset_end; i++ {
		if i >= len(*x)-1 {
			break
		}
		(*x_p)[k] = (*x)[i]
		(*y_p)[k] = (*y)[i]
		k += 1
	}
	(*learner).Fit(x_p, y_p)
	end := time.Now()
	fmt.Printf("start:%d end:%d %fsec\n", offset_start, offset_end, end.Sub(start).Seconds())
	return (*learner).GetParam()
}

func train(args []string) {
	var (
		model     string
		algorithm string
		eta       float64
		gamma     float64
		C         float64
		loop      int
		p         int
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
	fs.IntVar(&p, "p", 1, "number of processors")
	fs.Parse(args)

	//     fmt.Println(fs.Args())
	var (
		learner   gonline.LearnerInterface
		ftdict    gonline.Dict
		labeldict gonline.Dict
		x_train   [][]gonline.Feature
		y_train   []int
	)
	ftdict = gonline.NewDict()
	labeldict = gonline.NewDict()

	//     if p > 1 {
	if p > 0 {
		var wg sync.WaitGroup
		var numCpu int
		if numCpu > runtime.NumCPU() {
			numCpu = runtime.NumCPU()
			fmt.Println("number of processors is set to %d\n", numCpu)
		} else {
			numCpu = p
		}

		for _, trainfile := range fs.Args() {
			gonline.LoadTrainData(trainfile, &ftdict, &labeldict, &x_train, &y_train)
		}

		learners := make([]gonline.LearnerInterface, numCpu, numCpu)
		for i := 0; i < numCpu; i++ {
			switch algorithm {
			case "p":
				learners[i] = gonline.NewPerceptron()
			case "pa":
				learners[i] = gonline.NewPA("", C)
			case "pa1":
				learners[i] = gonline.NewPA("I", C)
			case "pa2":
				learners[i] = gonline.NewPA("II", C)
			case "cw":
				learners[i] = gonline.NewCW(eta)
			case "arow":
				learners[i] = gonline.NewArow(gamma)
			default:
				panic(fmt.Sprintf("Invalid algorithm: %s", algorithm))
			}
		}

		avg_weight := make([][]float64, 0, len(labeldict.Id2elem))
		var w *[][]float64
		semaphore := make(chan *[][]float64, numCpu)
		avg_num := len(x_train)/numCpu + 1
		fmt.Println("num of data for each learner", avg_num, "/", len(x_train))
		x_p := make([][]gonline.Feature, avg_num, avg_num)
		y_p := make([]int, avg_num, avg_num)
		for t := 0; t < loop; t++ {

			/* learn parameter for each learner independently */
			for i := 0; i < numCpu; i++ {
				wg.Add(1)
				go func(i int, learner *gonline.LearnerInterface) {
					defer wg.Done()
					semaphore <- update_parallel(i, avg_num, learner, &x_train, &y_train, &x_p, &y_p)
				}(i, &learners[i])
			}

			/* average parameters of leaners */
			for i := 0; i < numCpu; i++ {
				w = <-semaphore
				for lid, wvec := range *w {
					if len(avg_weight) <= lid {
						avg_weight = append(avg_weight, make([]float64, 0, 10))
					}
					for ftid, weight := range wvec {
						if len(avg_weight[lid]) <= ftid {
							for k := len(avg_weight[lid]); k <= ftid; k++ {
								avg_weight[lid] = append(avg_weight[lid], 0.)
							}
						}
						avg_weight[lid][ftid] += weight / float64(numCpu)
					}
				}
			}
			wg.Wait()

			/* share parameter */
			for i := 0; i < numCpu; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					learners[i].SetParam(&avg_weight)
				}(i)
			}
			wg.Wait()

			if testfile != "" {
				x_test := make([][]gonline.Feature, 0, 10000)
				y_test := make([]int, 0, 10000)
				gonline.LoadTestData(testfile, &ftdict, &labeldict, &x_test, &y_test)
				numCorr := 0
				numTotal := 0
				cls := gonline.Classifier{}
				cls.Weight = avg_weight
				for i, x_i := range x_test {
					j := cls.Predict(&x_i)
					if j == y_test[i] {
						numCorr += 1
					}
					numTotal += 1
				}
				acc1 := float64(numCorr) / float64(numTotal)
				fmt.Printf("epoch:%d numcpu:%d test accuracy: %f (%d/%d)\n", t+1, numCpu, acc1, numCorr, numTotal)
			}
			gonline.ShuffleData(&x_train, &y_train)
		}
	} else {
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
				gonline.LoadTrainData(trainfile, &ftdict, &labeldict, &x_train, &y_train)
				gonline.ShuffleData(&x_train, &y_train)
				learner.Fit(&x_train, &y_train)
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
				if j == y[i] {
					numCorr += 1
				}
				numTotal += 1
			}
		}
		acc := float64(numCorr) / float64(numTotal)
		fmt.Printf("train accuracy: %f (%d/%d)\n", acc, numCorr, numTotal)
	}
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
