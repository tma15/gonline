==
Passive Agressive II
==

Usage
==

```
Usage of ./pa:
-C=1: agressiveness parameter C > 0
-f="": data file
-l=10: number of iterations
-m="": mode {learn, test}
-v=false: verbose mode
-w="": model file
```

Train

```
./pa -f a1a -m learn -w model -l 1
```

Fit

```
./pa -f a1a.t -m test -w model
```
