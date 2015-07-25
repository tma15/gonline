package gonline

import (
	"bufio"
	"math"
	"os"
	"strconv"
	"strings"
)

type Classifier struct {
	Weight    [][]float64
	FtDict    Dict
	LabelDict Dict
}

func NewClassifier() Classifier {
	c := Classifier{
		Weight:    make([][]float64, 0, 100),
		FtDict:    NewDict(),
		LabelDict: NewDict(),
	}
	return c
}

func (this *Classifier) Predict(x *map[string]float64) int {
	argmax := -1
	max := math.Inf(-1)

	for labelid := 0; labelid < len(this.Weight); labelid++ {
		dot := 0.
		w := this.Weight[labelid]
		for ft, val := range *x {
			ftid := this.FtDict.Elem2id[ft]
			if ftid >= len(w) {
				continue
			}
			dot += w[ftid] * val
		}
		if dot > max {
			max = dot
			argmax = labelid
		}
	}
	return argmax
}

func LoadClassifier(fname string) Classifier {
	cls := NewClassifier()
	model_f, err := os.OpenFile(fname, os.O_RDONLY, 0644)
	if err != nil {
		panic("Failed to load model")
	}
	reader := bufio.NewReaderSize(model_f, 4096*32)
	line, err := reader.ReadString('\n')
	if err != nil {
		panic(err)
	}
	labelsize, _ := strconv.Atoi(strings.Trim(strings.Split(line, "\t")[1], "\n"))

	line, err = reader.ReadString('\n')
	ftsize, _ := strconv.Atoi(strings.Trim(strings.Split(line, "\t")[1], "\n"))
	for i := 0; i < labelsize; i++ {
		line, err = reader.ReadString('\n')
		cls.LabelDict.AddElem(strings.Trim(line, "\n"))
	}
	for i := 0; i < ftsize; i++ {
		line, err = reader.ReadString('\n')
		cls.FtDict.AddElem(strings.Trim(line, "\n"))
	}

	cls.Weight = make([][]float64, labelsize, labelsize)
	for labelid := 0; labelid < labelsize; labelid++ {
		cls.Weight[labelid] = make([]float64, ftsize, ftsize)
		for ftid := 0; ftid < ftsize; ftid++ {
			line, err = reader.ReadString('\n')
			line = strings.Trim(line, "\n")
			w, _ := strconv.ParseFloat(line, 64)
			cls.Weight[labelid][ftid] = w
		}
	}
	return cls
}
