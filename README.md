==
AdaGrad+RDA
==

Usage
==
---
adagrad
---
```
Usage of ./adagrad:
-c=0.1: regularization parameter
-f="": data file
-l=10: number of iterations
-m="": mode {learn, test}
-v=false: verbose mode
-w="": model file
```

---
Example
---

```
./adagrad -f news.train -m learn -w model -l 1 -c 0.01
./adagrad -f news.test -m test -w model -l 1 -c 0.01
```
