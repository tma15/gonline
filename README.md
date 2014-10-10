==
Perceptron
==

Options:

- f: data file
- m: mode {learn, test}
- w: model file
- l: number of iterations (default: 10)
- v: verbose mode

Learning
==
```
./perceptron -f=a1a -m=learn -w=model -l=100
```

Predicting
==
```
./perceptron -f=a1a.t -m=test -w=model
```
