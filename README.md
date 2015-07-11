==
Go Online Learner
==

Usage
==
To train learner:

```
$wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
$bunzip2 mnist.scale.bz2
$head -1000 mnist.scale > train.dat
$tail -1000 mnist.scale > test.dat
$./gonline train -a cw -m model -i 5 -t ./test.dat ./train.dat
testfile ./test
epoch:1 test accuracy: 0.828000 (828/1000)
epoch:2 test accuracy: 0.847000 (847/1000)
epoch:3 test accuracy: 0.844000 (844/1000)
epoch:4 test accuracy: 0.872000 (872/1000)
epoch:5 test accuracy: 0.857000 (857/1000)
train accuracy: 0.990000 (990/1000)
```

To evaluate learner:

```
./gonline test -m model test.dat
test accuracy: 0.857000 (857/1000)
```
