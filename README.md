==
Go Online Learner
==
A library of online learning algorithms written in golang.

Supported Algorithms
==
- Perceptron (p)
- Passive Aggressive (pa)
- Passive Aggressive I (pa1)
- Passive Aggressive II (pa2)
- Confidence Weighted (cw)
- AROW (arow)

Characters in parentheses are option arguments for `gonline train`.

Usage
==
To train learner:

```
$wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.bz2
$wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.bz2
$bunzip2 news20.bz2 news20.t.bz2
$./gonline train -a arow -m model -i 7 -t ./news20.t.scale ./news20.scale
algorithm: arow
testfile ./news20.t.scale
epoch:1 test accuracy: 0.770098 (3075/3993)
epoch:2 test accuracy: 0.795893 (3178/3993)
epoch:3 test accuracy: 0.802905 (3206/3993)
epoch:4 test accuracy: 0.804408 (3212/3993)
epoch:5 test accuracy: 0.800902 (3198/3993)
epoch:6 test accuracy: 0.800651 (3197/3993)
epoch:7 test accuracy: 0.803907 (3210/3993)
train accuracy: 0.977345 (15574/15935)
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
  -i=1: iteration number
  -m="": model filename
  -model="": model filename
  -t="": test file
```

To evaluate learner:

```
$./gonline test -m model news20.t.scale
test accuracy: 0.803907 (3210/3993)
```

Data Format
==
The format of training and testing data is:

```
<label> <feature1>:<value1> <feature2>:<value2> ...
```

Feature names such as `<feature1>` and `<feature2>` could be strings besides on integers. For example, words such as `soccer` and `baseball` can be used as `<feature1>` in text classification setting.
