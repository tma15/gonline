package gonline

import (
	"bufio"
	"fmt"
	"math"
	"os"
)

type PassiveAggressiveII struct {
	weight   [][]float64
	C        float64
	loop     int
	Labels   *Vocab
	Features *Vocab
}

func NewPassiveAggressiveII(C float64, loop int) PassiveAggressiveII {
	labels := NewVocab()
	features := NewVocab()
	p := PassiveAggressiveII{[][]float64{}, C, loop, labels, features}
	return p
}

func Norm(x []Fv) float64 {
	n := 0.
	for _, fv := range x {
		n += math.Pow(math.Abs(fv.V), 2)
	}
	return math.Sqrt(n)
}

func (p *PassiveAggressiveII) Update(X []Fv, yId int, sign float64) {
	for len(p.weight) < yId+1 {
		p.weight = append(p.weight, []float64{})
	}

	loss := math.Max(0, 1-sign*Dot(p.weight[yId], X))
	//         tau := loss / Norm(X)
	tau := loss / (Norm(X) + 1/(2*p.C))
	for i := 0; i < len(X); i++ {
		k := X[i].K
		for len(p.weight[yId]) < k+1 {
			p.weight[yId] = append(p.weight[yId], 0.)
		}
		p.weight[yId][k] += tau * sign
	}
}

func (p *PassiveAggressiveII) Predict(X []Fv) int {
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

func (p *PassiveAggressiveII) Fit(X [][]FvStr, y []string) {
	for loop := 0; loop < p.loop; loop++ {
		for i, X_i := range X {
			if _, ok := p.Labels.word2id[y[i]]; !ok {
				p.Labels.addWord(y[i])
			}
			yId_true := p.Labels.word2id[y[i]]
			x := fvstr2fv(p.Features, X_i, true)
			yId_pred := p.Predict(x)
			if yId_true != yId_pred {
				p.Update(x, yId_true, 1.)
				p.Update(x, yId_pred, -1.)
			}
		}
	}
}

func (p *PassiveAggressiveII) SaveModel(fname string) {
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
