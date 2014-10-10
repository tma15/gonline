package main

import (
	"fmt"
	"math"
)

type Perceptron struct {
	weight map[string]map[string]float64
	loop   int
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

func (p *Perceptron) InitWeigt(y []string) map[string]map[string]float64 {
	for _, y_i := range y {
		p.weight[y_i] = map[string]float64{}
	}
	return p.weight
}

func (p *Perceptron) Update(X map[string]float64, y string, sign float64) map[string]map[string]float64 {
	for f, _ := range X {
		_, ok := p.weight[y][f]
		if ok {
			p.weight[y][f] += sign
		} else {
			p.weight[y][f] = sign
		}
	}
	return p.weight
}

func (p *Perceptron) Predict(X map[string]float64) string {
	maxScore := math.Inf(-1)
	pred_y_i := ""
	for y_j, _ := range p.weight { // calculate scores for each label
		dot := Dot(X, p.weight[y_j])
		if dot > maxScore {
			maxScore = dot
			pred_y_i = y_j
		}
	}
	return pred_y_i
}

func (p *Perceptron) Fit(X []map[string]float64, y []string) map[string]map[string]float64 {
	p.weight = p.InitWeigt(y)
	for loop := 0; loop < p.loop; loop++ {
		fmt.Println(loop)
		for i, X_i := range X {
			pred_y_i := p.Predict(X_i)
			if y[i] != pred_y_i {
				p.Update(X_i, y[i], 1.)
				p.Update(X_i, pred_y_i, -1.)
			}
		}
	}
	return p.weight
}
