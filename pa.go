package gonline

import (
	"bufio"
	"fmt"
	//         "io"
	"math"
	"os"
	//         "strconv"
	//         "strings"
)

type PassiveAggressive struct {
	weight       Weight
	C            float64
	labelDefault string
	loop         int
}

func NewPassiveAggressive(C float64, loop int) PassiveAggressive {
	p := PassiveAggressive{Weight{}, C, "", loop}
	return p
}

func Norm(x map[string]float64) float64 {
	n := 0.
	for _, v := range x {
		n += math.Pow(math.Abs(v), 2)
	}
	return math.Sqrt(n)
}

func (p *PassiveAggressive) InitWeigt(y []string) Weight {
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
		if cnt > maxCnt {
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

func (p *PassiveAggressive) Update(X map[string]float64, y string, sign float64) Weight {
	loss := math.Max(0, 1-sign*Dot(X, p.weight[y]))
	//         tau := loss / Norm(X)
	tau := loss / (Norm(X) + 1/(2*p.C))
	if _, ok := p.weight[y]; ok == false {
		p.weight[y] = map[string]float64{}
	}

	for f, _ := range X {
		if _, ok := p.weight[y][f]; ok {
			p.weight[y][f] += tau * sign
		} else {
			p.weight[y][f] = tau * sign
		}
	}
	return p.weight
}

func (p *PassiveAggressive) haveSameScores(scores []float64) bool {
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

func (p *PassiveAggressive) Predict(X map[string]float64) string {
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
	//                 fmt.Println(scores)
	//                 fmt.Println("pred y", pred_y_i, "Default", p.labelDefault)
	//                 fmt.Println("")
	return pred_y_i
}

func (p *PassiveAggressive) Fit(X []map[string]float64, y []string) Weight {
	p.weight = p.InitWeigt(y)
	for loop := 0; loop < p.loop; loop++ {
		//                 fmt.Println(loop)
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

func (p *PassiveAggressive) SaveModel(fname string) {
	model_f, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("Failed to dump model")
	}

	writer := bufio.NewWriterSize(model_f, 4096*32)
	writer.WriteString("DEFAULTLABEL\n")
	writer.WriteString(fmt.Sprintf("%s\n", p.labelDefault))
	//         writer.WriteString("C\n")
	//         writer.WriteString(fmt.Sprintf("%f\n", p.C))
	writer.WriteString("WEIGHT\n")
	for y, weight := range p.weight {
		for ft, w := range weight {
			writer.WriteString(fmt.Sprintf("%s\t%s\t%f\n", y, ft, w))
		}
	}
	writer.Flush()
}
