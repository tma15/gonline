package gonline

import (
	"bufio"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

type Classifier struct {
	w            Weight
	labelDefault string
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

func (c *Classifier) haveSameScores(scores []float64) bool {
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

func (c *Classifier) Predict(X map[string]float64) (string, float64) {
	maxScore := math.Inf(-1)
	pred_y_i := ""
        scores := []float64{}
	for y_j, _ := range c.w { // calculate scores for each label
		dot := Dot(X, c.w[y_j])
		scores = append(scores, dot)
		if dot > maxScore {
			maxScore = dot
			pred_y_i = y_j
		}
	}
	if c.haveSameScores(scores) {
		pred_y_i = c.labelDefault
	}
	return pred_y_i, maxScore
}

func LoadClassifier(fname string) Classifier {
	model_f, err := os.OpenFile(fname, os.O_RDONLY, 0644)
	if err != nil {
		panic("Failed to load model")
	}
	weight := Weight{}
	var labelDefault string
	reader := bufio.NewReaderSize(model_f, 4096*32)
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
				weight[label][id] = w
			}
		}
		if a == "d" {
			labelDefault = text
		}

	}
	cls := Classifier{weight, labelDefault}
//         os.Exit(1)
	return cls
}
