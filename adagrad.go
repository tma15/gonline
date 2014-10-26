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

type AdaGrad struct {
	weight       Weight
	lambda       float64
	labelDefault string
	loop         int
	gradSum      map[string]map[string]float64
	H            map[string]map[string]float64
}

func NewAdaGrad(lambda float64, loop int) AdaGrad {
	p := AdaGrad{Weight{}, lambda, "", loop,
		map[string]map[string]float64{},
		map[string]map[string]float64{}}
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

func (p *AdaGrad) InitWeigt(y []string) Weight {
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

func (p *AdaGrad) Update(X map[string]float64, y string, sign_y, t int) Weight {
	if _, ok := p.gradSum[y]; ok == false {
		p.gradSum[y] = map[string]float64{}
	}
	if _, ok := p.H[y]; ok == false {
		p.H[y] = map[string]float64{}
	}

	var sign_g float64
	for f, _ := range X {
		grad_i := float64(sign_y) * X[f]
		if grad_i == 0 {
			continue
		}

		if _, ok := p.gradSum[y][f]; ok {
			p.gradSum[y][f] += grad_i
		} else {
			p.gradSum[y][f] = grad_i
		}
		gradMean_f := p.gradSum[y][f] / float64(t)

		if gradMean_f > 0 {
			sign_g = 1.
		} else {
			sign_g = -1.
		}

		if _, ok := p.H[y][f]; ok {
			p.H[y][f] += grad_i * grad_i
		} else {
			p.H[y][f] = grad_i * grad_i
		}
		w := -1.0 * sign_g * float64(t) * float64(math.Max(0, math.Abs(gradMean_f)-p.lambda)/math.Sqrt(p.H[y][f]))
		p.weight[y][f] = w
	}
	return p.weight
}

func (p *AdaGrad) haveSameScores(scores []float64) bool {
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

func (p *AdaGrad) Predict(X map[string]float64) string {
	maxScore := math.Inf(-1)
	pred_y_i := ""
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
		pred_y_i = p.labelDefault
	}
//         fmt.Println(scores)
//         fmt.Println("predy", pred_y_i, "Default", p.labelDefault)
//         fmt.Println("")
	return pred_y_i
}

func (p *AdaGrad) Fit(X []map[string]float64, y []string) Weight {
	p.weight = p.InitWeigt(y)
	var t int

	for loop := 0; loop < p.loop; loop++ {
		fmt.Println(loop)
		for i, X_i := range X {
			t += 1
			y_j := p.Predict(X_i)
			if y_j != y[i] {
				p.Update(X_i, y_j, 1, t)
				p.Update(X_i, y[i], -1, t)
			}
		}
	}
	return p.weight
}

func SaveModel(p AdaGrad, fname string) {
	model_f, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("Failed to dump model")
	}

	writer := bufio.NewWriterSize(model_f, 4096*32)
	writer.WriteString("DEFAULTLABEL\n")
	writer.WriteString(fmt.Sprintf("%s\n", p.labelDefault))
	writer.WriteString("WEIGHT\n")
	for y, weight := range p.weight {
		for ft, w := range weight {
			writer.WriteString(fmt.Sprintf("%s\t%s\t%f\n", y, ft, w))
		}
	}
	writer.Flush()
}

func LoadModel(fname string) AdaGrad {
	model_f, err := os.OpenFile(fname, os.O_RDONLY, 0644)
	if err != nil {
		panic("Failed to dump model")
	}
	weight := Weight{}
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
		if a == "d" {
			labelDefault = text
		}

	}
	p := AdaGrad{weight, 0, labelDefault, 0,
		map[string]map[string]float64{},
		map[string]map[string]float64{}}
	return p
}
