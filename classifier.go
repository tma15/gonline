package gonline

import (
	"bufio"
	"fmt"
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

func (this *Classifier) conv(x *map[string]float64) *[]Feature {
	num_feat := len(*x)
	features := make([]Feature, 0, num_feat)
	for name, val := range *x {
		if !this.FtDict.HasElem(name) {
			continue
		}
		id := this.FtDict.Elem2id[name]
		f := NewFeature(id, val, name)
		features = append(features, f)
	}
	return &features
}

func (this *Classifier) Predict(x *map[string]float64) int {
	argmax := -1
	max := math.Inf(-1)
	features := this.conv(x)
	for labelid := 0; labelid < len(this.Weight); labelid++ {
		dot := 0.
		w := this.Weight[labelid]
		for _, f := range *features {
			if f.Id >= len(w) { /* weight of this feature is zero. */
				continue
			}
			dot += w[f.Id] * f.Val
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
	for _, m := range margins[:n] {
		topn[i] = m.Id
		topnmargins[i] = m.Val
		i++
	}
	return topn, topnmargins

}

func LoadClassifier(fname string) Classifier {
	cls := NewClassifier()
	model_f, err := os.OpenFile(fname, os.O_RDONLY, 0644)
	defer model_f.Close()
	if err != nil {
		panic(fmt.Sprintf("Failed to load model:%s", fname))
	}
	reader := bufio.NewReaderSize(model_f, 4096*32)
	line, err := reader.ReadString('\n')
	if err != nil {
		panic(err)
	}
	labelsize, _ := strconv.Atoi(strings.Trim(strings.Split(line, "\t")[1], "\n"))

	for i := 0; i < labelsize; i++ {
		line, err = reader.ReadString('\n')
		cls.LabelDict.AddElem(strings.Trim(line, "\n"))
	}
	cls.Weight = make([][]float64, labelsize, labelsize)
	for labelid := 0; labelid < labelsize; labelid++ {
		line, err = reader.ReadString('\n')
		line = strings.Trim(line, "\n")
		ftsize, _ := strconv.Atoi(strings.Trim(strings.Split(line, "\t")[1], "\n"))
		cls.Weight[labelid] = make([]float64, ftsize, ftsize)
		for i := 0; i < ftsize; i++ {
			line, err = reader.ReadString('\n')
			line = strings.Trim(line, "\n")
			sp := strings.Split(line, " ")
			ft := sp[0]
			w, err := strconv.ParseFloat(sp[1], 64)
			if err != nil {
				panic(err)
			}
			if !cls.FtDict.HasElem(ft) {
				cls.FtDict.AddElem(ft)
			}
			ftid := cls.FtDict.Elem2id[ft]
			cls.Weight[labelid][ftid] = w
		}
	}
	return cls
}
