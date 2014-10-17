package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func LoadFromStdin() ([]map[string]float64, []string) {
	scanner := bufio.NewScanner(os.Stdin)
	X := []map[string]float64{}
	y := []string{}
	for scanner.Scan() {
		fv := strings.SplitN(scanner.Text(), " ", 2)
		y_i := fv[0]
		x := map[string]float64{}
		for _, k := range strings.Split(strings.Trim(fv[1], " "), " ") {
			i := strings.Split(k, ":")
			i64, _ := strconv.ParseFloat(i[1], 64)
			x[i[0]] = i64
		}
		X = append(X, x)
		y = append(y, y_i)
	}
	return X, y
}

func LoadFromFile(fname string) ([]map[string]float64, []string) {
	fp, err := os.Open(fname)
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(fp)
	X := []map[string]float64{}
	y := []string{}
	for scanner.Scan() {
		//                 fmt.Println(scanner.Text())
		fv := strings.SplitN(scanner.Text(), " ", 2)
		y_i := fv[0]
		x := map[string]float64{}
		for _, k := range strings.Split(strings.Trim(fv[1], " "), " ") {
			i := strings.Split(k, ":")
			i64, _ := strconv.ParseFloat(i[1], 64)
			x[i[0]] = i64
		}
		X = append(X, x)
		y = append(y, y_i)
	}
	return X, y

}


func confusionMatrix(y, pred_y []string) {
	if len(y) != len(pred_y) {
		panic("Numbers of labels must be the same")
	}

	confMat := map[string]map[string]int{}
	for i, y_i := range y {
		if _, ok := confMat[y_i]; ok == false {
			confMat[y_i] = map[string]int{}
		}
		if _, ok := confMat[y_i][pred_y[i]]; ok {
			confMat[y_i][pred_y[i]] += 1
		} else {
			confMat[y_i][pred_y[i]] = 1
		}
	}

	var prec float64
	var recall float64
	for k, _ := range confMat {
		b := 0
		a := 0
		c := 0
		d := 0
		for k_i, _ := range confMat[k] {
			if k == k_i {
				a = confMat[k][k_i]
				c = confMat[k_i][k]
			}
			b += confMat[k][k_i]
			d += confMat[k_i][k]
		}
		recall = float64(a) / float64(b)
		prec = float64(c) / float64(d)

		o1 := fmt.Sprintf("Recall[%s]: %f (%d/%d)", k, recall, a, b)
		o2 := fmt.Sprintf("Prec[%s]: %f (%d/%d)", k, prec, c, d)
		fmt.Println(o1)
		fmt.Println(o2)
		fmt.Println("--")
	}
}
