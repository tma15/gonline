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
epoch:1 test accuracy: 0.821438 (3280/3993)
epoch:2 test accuracy: 0.835212 (3335/3993)
epoch:3 test accuracy: 0.842725 (3365/3993)
epoch:4 test accuracy: 0.845980 (3378/3993)
epoch:5 test accuracy: 0.849236 (3391/3993)
epoch:6 test accuracy: 0.853243 (3407/3993)
epoch:7 test accuracy: 0.854746 (3413/3993)
epoch:8 test accuracy: 0.856749 (3421/3993)
epoch:9 test accuracy: 0.859254 (3431/3993)
epoch:10 test accuracy: 0.859755 (3433/3993)
./gonline train -a arow -m model -i 10 -t ./news20.t.scale -withoutshuffle   109.53s user 1.65s system 98% cpu 1:53.25 total
```

In practice, shuffling training data can improve accuracy.

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
test accuracy: 0.859755 (3433/3993)
```

Benchmark
==
For all algorithms which are supported by `gonline`, fitting 1 iteration on training data `news.scale`, then predicting test data `news.t.scale`. Training data don't be shuffled. Default values are used as hyper parameters.

|algorithm|accuracy|
|---------|--------|
|Perceptron|0.758327|
|Passive Aggressive|0.745805|
|Passive Aggressive I|0.758327|
|Passive Aggressive II|0.757576|
|Confidence Weighted|0.784874|
|AROW (the top-1 version)|0.782870|
|AROW (the full version)|0.859755|

AROW (the top-1 version) is not supported now.
Evaluation is conducted using following command:
```
$./gonline train -a <ALGORITHM> -m model -i 1 -t ./news20.t.scale -withoutshuffle ./news20.scale
```

Accuracy of SVMs with linear kernel which is supported by `libsvm`:
```
$svm-train -t 0 news20.scale
$svm-predict news20.t.scale news20.scale.model out
Accuracy = 84.022% (3355/3993) (classification)
```

TODO: Hyper parameters tuning for each algorithm using development data.

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
- Koby Crammer, Alex Kulesza, and Mark Dredze. "Adaptive Regularization of Weight Vectors". Machine Learning. 2013.

Licence
==
This software is released under the MIT License, see LICENSE.txt.
