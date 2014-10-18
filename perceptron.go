package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

type Weight map[string]map[string]float64

type Perceptron struct {
	weight       Weight
	eta          float64
	labelDefault string
	loop         int
}

func NewPerceptron(eta float64, loop int) Perceptron {
	p := Perceptron{Weight{}, eta, "", loop}
	return p
}

func Dot(v, w map[string]float64) float64 {
	dot := 0.
	for key, value := range v {
		_, ok := w[key]
		if ok {
			dot += value * w[key]
		}
	}
	return dot
}

func (p *Perceptron) InitWeigt(y []string) Weight {
	labelDist := map[string]int{}
	for _, y_i := range y {
		p.weight[y_i] = map[string]float64{}
		if _, ok := labelDist[y_i]; ok {
			labelDist[y_i] += 1
		} else {
			labelDist[y_i] = 1
		}
	}

	maxCnt := -1
	l := ""
	for k, cnt := range labelDist {
		//                 fmt.Println(k, cnt)
		if cnt > maxCnt {
			//                     fmt.Println("Default:", k)
			p.labelDefault = k
			maxCnt = cnt
		}
		l = k
	}
	if p.labelDefault == "" {
		p.labelDefault = l
	}

	return p.weight
}

func (p *Perceptron) Update(X map[string]float64, y string, sign float64) Weight {
	for f, _ := range X {
		_, ok := p.weight[y][f]
		if ok {
			p.weight[y][f] += p.eta * sign
		} else {
			p.weight[y][f] = p.eta * sign
		}
	}
	return p.weight
}

func (p *Perceptron) haveSameScores(scores []float64) bool {
	n_t := 0
	var prev float64
	for i, score := range scores {
		if i == 0 {
			prev = score
		} else {
			if score == prev { // current score is the same with previous score
				n_t += 1
			}
			prev = score
		}
	}
	if n_t == len(scores)-1 {
		return true
	} else {
		return false
	}
}

func (p *Perceptron) Predict(X map[string]float64) string {
	maxScore := math.Inf(-1)
	pred_y_i := ""
	//         allSame := true
	scores := []float64{}
	for y_j, _ := range p.weight { // calculate scores for each label
		dot := Dot(X, p.weight[y_j])
		scores = append(scores, dot)
		//                 fmt.Println(y_j, dot)
		if dot > maxScore {
			maxScore = dot
			pred_y_i = y_j
		}
	}
	if p.haveSameScores(scores) {
		//                 fmt.Println("aaa")
		pred_y_i = p.labelDefault
	}
	//         fmt.Println(scores)
	//         fmt.Println("pred y", pred_y_i, "Default", p.labelDefault)
	//         fmt.Println("")
	return pred_y_i
}

func (p *Perceptron) Fit(X []map[string]float64, y []string) Weight {
	p.weight = p.InitWeigt(y)
	for loop := 0; loop < p.loop; loop++ {
		fmt.Println(loop)
		for i, X_i := range X {
			//                         fmt.Println(i, X_i)
			pred_y_i := p.Predict(X_i)
			if y[i] != pred_y_i {
				p.Update(X_i, y[i], 1.)
				p.Update(X_i, pred_y_i, -1.)
			}
		}
	}
	return p.weight
}

func SaveModel(p Perceptron, fname string) {
	model_f, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("Failed to dump model")
	}

	writer := bufio.NewWriterSize(model_f, 4096*32)
	writer.WriteString("DEFAULTLABEL\n")
	writer.WriteString(fmt.Sprintf("%s\n", p.labelDefault))
	writer.WriteString("ETA\n")
	writer.WriteString(fmt.Sprintf("%f\n", p.eta))
	writer.WriteString("WEIGHT\n")
	for y, weight := range p.weight {
		for ft, w := range weight {
			writer.WriteString(fmt.Sprintf("%s\t%s\t%f\n", y, ft, w))
		}
	}
	writer.Flush()
}

func LoadModel(fname string) Perceptron {
	model_f, err := os.OpenFile(fname, os.O_RDONLY, 0644)
	if err != nil {
		panic("Failed to dump model")
	}
	weight := Weight{}
	var eta float64
	var labelDefault string
	reader := bufio.NewReaderSize(model_f, 4096)
	a := ""
	for {
		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		text := string(line)
		if text == "WEIGHT" {
			a = "w"
			continue
		}
		if text == "ETA" {
			a = "e"
			continue
		}
		if text == "DEFAULTLABEL" {
			a = "d"
			continue
		}
		if a == "w" {
			elems := strings.Split(text, "\t")
			label := elems[0]
			id := elems[1]
			w, _ := strconv.ParseFloat(elems[2], 64)
			_, ok := weight[label]
			if ok {
				weight[label][id] = w
			} else {
				weight[label] = map[string]float64{}
			}
		}
		if a == "e" {
			eta, _ = strconv.ParseFloat(text, 64)
		}
		if a == "d" {
			labelDefault = text
		}

	}
	p := Perceptron{weight, eta, labelDefault, 0}
	return p
}
