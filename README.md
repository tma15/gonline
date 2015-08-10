==
gonline
==
A library of online machine learning algorithms written in golang.

How to Install
==
```
$go get github.com/tma15/gonline
```

How to Build
==
```
$cd $GOPATH/src/github.com/tma15/gonline/gonline
$go build
```

Supported Algorithms
==
- Perceptron (p)
- Passive Aggressive (pa)
- Passive Aggressive I (pa1)
- Passive Aggressive II (pa2)
- Confidence Weighted (cw)
- AROW (arow)

Characters in parentheses are option arguments for `-a` of `gonline train`.

Usage
==
Template command of training:
```
$./gonline train -a <ALGORITHM> -m <MODELFILE> -t <TESTINGFILE> -i <ITERATION> <TRAININGFILE1> <TRAININGFILE2> ... <TRAININGFILEK>
```

To train learner by AROW algorithm:
```
$wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.scale.bz2
$wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.scale.bz2
$bunzip2 news20.scale.bz2 news20.t.scale.bz2
$time ./gonline train -a arow -m model -i 10 -t ./news20.t.scale -withoutshuffle ./news20.scale
algorithm: AROW
testfile ./news20.t.scale
training data will not be shuffled
epoch:1 test accuracy: 0.782870 (3126/3993)
epoch:2 test accuracy: 0.798147 (3187/3993)
epoch:3 test accuracy: 0.797646 (3185/3993)
epoch:4 test accuracy: 0.801402 (3200/3993)
epoch:5 test accuracy: 0.802655 (3205/3993)
epoch:6 test accuracy: 0.804909 (3214/3993)
epoch:7 test accuracy: 0.805910 (3218/3993)
epoch:8 test accuracy: 0.806411 (3220/3993)
epoch:9 test accuracy: 0.807663 (3225/3993)
epoch:10 test accuracy: 0.807663 (3225/3993)
./gonline train -a arow -m model -i 10 -t ./news20.t.scale -withoutshuffle   69.32s user 1.24s system 98% cpu 1:11.43 total
```

You can see more command options using help option:

```
$./gonline train -h
Usage of train:
  -C=0.01: degree of aggressiveness for PA-I and PA-II
  -a="": algorithm for training {p, pa, pa1, pa2, cw, arow}
  -algorithm="": algorithm for training {p, pa, pa1, pa2, cw, arow}
  -eta=0.8: confidence parameter for Confidence Weighted
  -g=10: regularization parameter for AROW
  -i=1: number of iterations
  -m="": file name of model
  -model="": file name of model
  -t="": file name of test data
  -withoutshuffle=false: doesn't shuffle the training data
```

Template command of testing:
```
$./gonline test -m <MODELFILE> <TESTINGFILE1> <TESTINGFILE2> ... <TESTINGFILEK>
```

To evaluate learner:

```
$./gonline test -m model news20.t.scale
test accuracy: 0.807663 (3225/3993)
```

Data Format
==
The format of training and testing data is:

```
<label> <feature1>:<value1> <feature2>:<value2> ...
```

Feature names such as `<feature1>` and `<feature2>` could be strings besides on integers. For example, words such as `soccer` and `baseball` can be used as `<feature1>` in text classification setting.

References
==
- Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz and Yoram Singer. "Online Passive-Aggressive Algorithms". JMLR. 2006.
- Mark Dredze, Koby Crammer and Fernando Pereira. "Confidence-Weighted Linear Classification". ICML. 2008.
- Koby Crammer, Mark Dredze and Alex Kulesza. "Multi-Class Confidence Weighted Algorithms". EMNLP. 2009.
- Koby Crammer, Alex Kulesza and Mark Dredze. "Adaptive Regularization of Weight Vectors". NIPS. 2009.

Licence
==
This software is released under the MIT License, see LICENSE.txt.
