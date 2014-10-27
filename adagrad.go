package gonline

import (
	"bufio"
	"fmt"
	"math"
	"os"
)

type AdaGrad struct {
	weight   [][]float64
	lambda   float64
	loop     int
	gradSum  [][]float64
	H        [][]float64
	Labels   *Vocab
	Features *Vocab
}

func NewAdaGrad(lambda float64, loop int) AdaGrad {
	labels := NewVocab()
	features := NewVocab()
	p := AdaGrad{[][]float64{}, lambda, loop,
		[][]float64{},
		[][]float64{},
		labels,
		features,
	}
	return p
}

func (p *AdaGrad) Update(X []Fv, yId int, sign_y, t int) {
	for len(p.weight) < yId+1 {
		p.weight = append(p.weight, []float64{})
	}
	for len(p.gradSum) < yId+1 {
		p.gradSum = append(p.gradSum, []float64{})
	}
	for len(p.H) < yId+1 {
		p.H = append(p.H, []float64{})
	}

	var sign_g float64
	for _, fv := range X {
		grad_i := float64(sign_y) * fv.V
		for len(p.weight[yId]) < fv.K+1 {
			p.weight[yId] = append(p.weight[yId], 0.)
		}

		for len(p.gradSum[yId]) < fv.K+1 {
			p.gradSum[yId] = append(p.gradSum[yId], 0.)
		}
		p.gradSum[yId][fv.K] += -1. * grad_i
		gradMean_f := p.gradSum[yId][fv.K] / float64(t)

		for len(p.H[yId]) < fv.K+1 {
			p.H[yId] = append(p.H[yId], 0.)
		}
		p.H[yId][fv.K] += grad_i * grad_i

		if gradMean_f > 0 {
			sign_g = 1.
		} else {
			sign_g = -1.
		}
		w := -1.0 * sign_g * float64(t) * float64(math.Max(0, math.Abs(gradMean_f)-p.lambda)/math.Sqrt(p.H[yId][fv.K]))
		p.weight[yId][fv.K] = w
	}
}

func (p *AdaGrad) Predict(X []Fv) int {
	maxScore := math.Inf(-1)
	var maxyId int
	scores := []float64{}
	for j, w := range p.weight { // calculate scores for each label
		dot := Dot(w, X)
		scores = append(scores, dot)
		if dot > maxScore {
			maxScore = dot
			maxyId = j
		}
	}
	return maxyId
}

func (p *AdaGrad) Fit(X [][]FvStr, y []string) {
	t := 0
	for loop := 0; loop < p.loop; loop++ {
		for i, X_i := range X {
			t += 1
			if _, ok := p.Labels.word2id[y[i]]; !ok {
				p.Labels.addWord(y[i])
			}
			yId_true := p.Labels.word2id[y[i]]
			x := fvstr2fv(p.Features, X_i, true)
			yId_pred := p.Predict(x)
			if yId_true != yId_pred {
				p.Update(x, yId_true, 1., t)
				p.Update(x, yId_pred, -1., t)
			}
		}
	}
}

func (p *AdaGrad) SaveModel(fname string) {
	model_f, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("Failed to dump model")
	}

	writer := bufio.NewWriterSize(model_f, 4096*32)
	for yId, weight := range p.weight {
		for ftId, w := range weight {
			y := p.Labels.words[yId]
			ft := p.Features.words[ftId]
			writer.WriteString(fmt.Sprintf("%s\t%s\t%f\n", y, ft, w))
		}
	}
	writer.Flush()
}
