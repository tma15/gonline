package gonline

import (
	"bufio"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

type Margin struct {
	Id  int
	Val float64
}

type Margins []Margin

func (this Margins) Len() int {
	return len(this)
}

func (this Margins) Less(i, j int) bool {
	if this[i].Val < this[j].Val {
		return true
	} else {
		return false
	}
}

func (this Margins) Swap(i, j int) {
	this[i], this[j] = this[j], this[i]
}

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
			if !this.FtDict.HasElem(ft) {
				continue
			}
			ftid := this.FtDict.Elem2id[ft]
			dot += w[ftid] * val
		}
		if dot > max {
			max = dot
			argmax = labelid
		}
	}
	return argmax
}

func (this *Classifier) PredictTopN(x *map[string]float64, n int) ([]int, []float64) {
	margins := Margins{}

	for labelid := 0; labelid < len(this.Weight); labelid++ {
		dot := 0.
		w := this.Weight[labelid]
		for ft, val := range *x {
			if !this.FtDict.HasElem(ft) {
				continue
			}
			ftid := this.FtDict.Elem2id[ft]
			dot += w[ftid] * val
		}
		margins = append(margins, Margin{Id: labelid, Val: dot})
	}
	sort.Sort(sort.Reverse(margins))
	topn := make([]int, n, n)
	topnmargins := make([]float64, n, n)
	i := 0
	for _, m := range margins {
		topn[i] = m.Id
		topnmargins[i] = m.Val
		i++
	}
	return topn, topnmargins

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
