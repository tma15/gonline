package gonline

import (
	"bufio"
	"math"
	"os"
	"strconv"
	"strings"
)

type Classifier struct {
	Weight [][]float64
}

func (this *Classifier) Predict(x *[]Feature) int {
	argmax := -1
	max := math.Inf(-1)

	for labelid := 0; labelid < len(this.Weight); labelid++ {
		dot := 0.
		w := this.Weight[labelid]
		for _, ft := range *x {
			dot += w[ft.Id] * ft.Val
		}
		if dot > max {
			max = dot
			argmax = labelid
		}
	}
	return argmax
}

func LoadClassifier(fname string, ftdict, labeldict *Dict) Classifier {
	model_f, err := os.OpenFile(fname, os.O_RDONLY, 0644)
	if err != nil {
		panic("Failed to load model")
	}
	var (
		cls Classifier
	)
	reader := bufio.NewReaderSize(model_f, 4096*32)
	line, err := reader.ReadString('\n')
	if err != nil {
		panic(err)
	}
	labelsize, _ := strconv.Atoi(strings.Trim(strings.Split(line, "\t")[1], "\n"))

	line, err = reader.ReadString('\n')
	ftsize, _ := strconv.Atoi(strings.Trim(strings.Split(line, "\t")[1], "\n"))
	//     fmt.Println(labelsize, ftsize)
	for i := 0; i < labelsize; i++ {
		line, err = reader.ReadString('\n')
		(*labeldict).AddElem(strings.Trim(line, "\n"))
	}
	for i := 0; i < ftsize; i++ {
		line, err = reader.ReadString('\n')
		(*ftdict).AddElem(strings.Trim(line, "\n"))
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
