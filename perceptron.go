package gonline

import (
	"bufio"
	"fmt"
	"math"
	"os"
)


type Perceptron struct {
	weight       [][]float64
	eta          float64
	labelDefault string
	loop         int
	labels       *Vocab
	features     *Vocab
}

func NewPerceptron(eta float64, loop int) Perceptron {
	labels := NewVocab()
	features := NewVocab()
	p := Perceptron{[][]float64{}, eta, "", loop, labels, features}
	return p
}

func (p *Perceptron) Update(X []Fv, yId int, sign float64) {
	for len(p.weight) < yId+1 {
		p.weight = append(p.weight, []float64{})
	}
	for i := 0; i < len(X); i++ {
		k := X[i].K // feature id
		for len(p.weight[yId]) < k+1 {
			p.weight[yId] = append(p.weight[yId], 0.) // expand weight vector
		}
		p.weight[yId][k] += p.eta * sign
	}
}

func (p *Perceptron) Predict(X []Fv) int {
	maxScore := math.Inf(-1)
	var maxyId int
	scores := []float64{}
	for j, w := range p.weight { // calculate scores for each label
		dot := Dot(w, X)
		//                 fmt.Println(dot)
		scores = append(scores, dot)
		if dot > maxScore {
			maxScore = dot
			maxyId = j
		}
	}
	return maxyId
}

func (p *Perceptron) Fit(X [][]FvStr, y []string) {
	for loop := 0; loop < p.loop; loop++ {
		for i, X_i := range X {
			if _, ok := p.labels.word2id[y[i]]; !ok {
				p.labels.addWord(y[i])
			}
			yId_true := p.labels.word2id[y[i]]
			x := fvstr2fv(p.features, X_i, true)
			yId_pred := p.Predict(x)
			if yId_true != yId_pred {
				p.Update(x, yId_true, 1.)
				p.Update(x, yId_pred, -1.)
			}
		}
	}
}

func (p *Perceptron) SaveModel(fname string) {
	model_f, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("Failed to dump model")
	}

	writer := bufio.NewWriterSize(model_f, 4096*32)
	for yId, weight := range p.weight {
		for ftId, w := range weight {
			y := p.labels.words[yId]
			ft := p.features.words[ftId]
			writer.WriteString(fmt.Sprintf("%s\t%s\t%f\n", y, ft, w))
		}
	}
	writer.Flush()
}
