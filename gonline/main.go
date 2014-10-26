package main

import (
	"flag"
	"fmt"
        "github.com/tma15/gonline"
	"os"
)

func train(args []string) {
    var (
        model string
        algorithm string
        eta float64
        c float64
        loop int
    )
//     fmt.Println(args)

    fs := flag.NewFlagSet("train", flag.ExitOnError)
    fs.StringVar(&model,  "model", "", "model filename")
    fs.StringVar(&model,  "m", "", "model filename")
    fs.StringVar(&algorithm,  "algorithm", "", "algorithm for training {perceptron, pa2, adagrad")
    fs.StringVar(&algorithm,  "a", "", "algorithm for training {perceptron, pa2, adagrad")
    fs.Float64Var(&eta,  "eta", 0.1, "learning rate")
    fs.Float64Var(&c,  "c", 1e-5, "regularization parameter")
    fs.IntVar(&loop,  "i", 1, "iteration number")
    fs.Parse(args)

//     fmt.Println(fs.Args())
    var p interface{}
    switch algorithm {
    case "perceptron":
        p = gonline.NewPerceptron(eta, loop)
        cls, _ := p.(gonline.Perceptron)
        for _, trainfile := range fs.Args() {
            X, y := gonline.LoadFromFile(trainfile)
            cls.Fit(X, y)
        }
        cls.SaveModel(model)
    case "pa2":
        p = gonline.NewPassiveAggressive(c, loop)
        cls, _ := p.(gonline.PassiveAggressive)
        for _, trainfile := range fs.Args() {
            X, y := gonline.LoadFromFile(trainfile)
            cls.Fit(X, y)
        }
        cls.SaveModel(model)
    case "adagrad":
        p = gonline.NewAdaGrad(c, loop)
        cls, _ := p.(gonline.AdaGrad)
        for _, trainfile := range fs.Args() {
            X, y := gonline.LoadFromFile(trainfile)
            cls.Fit(X, y)
        }
        cls.SaveModel(model)
    default:
        panic("Invalid algorithm")
    }
}

func test(args []string) {
    var (
        model string
    )
    fs := flag.NewFlagSet("test", flag.ExitOnError)
    fs.StringVar(&model,  "model", "", "model filename")
    fs.StringVar(&model,  "m", "", "model filename")
    fs.Parse(args)


    cls := gonline.LoadClassifier(model)

    for _, trainfile := range fs.Args() {
        X, y := gonline.LoadFromFile(trainfile)
        predy := make([]string, len(X))
        for i, x_i := range X {
            y_i_pred, _ := cls.Predict(x_i)
            predy[i] = y_i_pred
//             fmt.Println(i, y_i_pred, score)
        }
        gonline.ConfusionMatrix(y, predy)
    }

}

var usage = `
Usage of %s <Command> [Options]

Commands:
  train
  test

`

func main() {
        flag.Usage = func () {
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
