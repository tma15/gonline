package gonline

import (
	"bufio"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

type Fv struct {
	K int
	V float64
}

type FvStr struct {
	K string
	V float64
}

func fvstr2fv(v *Vocab, fvstr []FvStr, update bool) []Fv {
	fv := make([]Fv, 0, len(fvstr))
	for _, f := range fvstr {
		k, ok := v.word2id[f.K]
		if ok {
			fv = append(fv, Fv{k, f.V})
		} else if update {
			k = v.addWord(f.K)
			fv = append(fv, Fv{k, f.V})
		}
	}
	return fv
}

type Vocab struct {
	words   []string
	word2id map[string]int
}

func NewVocab() *Vocab {
	v := Vocab{[]string{}, map[string]int{}}
	return &v
}

func (v *Vocab) addWord(word string) int {
	newId := len(v.words)
	v.words = append(v.words, word)
	v.word2id[word] = newId
	return newId
}

func (v *Vocab) getId(word string, update bool) int {
	if id, ok := v.word2id[word]; ok {
		return id
	}

	if update {
		return v.addWord(word)
	}
	return -1
}

type Classifier struct {
	weight   [][]float64
	Labels   *Vocab
	Features *Vocab
}

func Dot(w []float64, fv []Fv) float64 {
	dot := 0.
	for i := 0; i < len(fv); i++ {
		k := fv[i].K      // feature id
		if len(w) < k+1 { // Out of vocabulary
			continue
		}
		dot += w[k] * fv[i].V
	}
	return dot
}

func (cls *Classifier) Predict(X []FvStr) (string, float64) {
	x := fvstr2fv(cls.Features, X, false)
	maxScore := math.Inf(-1)
	var maxyId int
	scores := []float64{}
	for j, w := range cls.weight { // calculate scores for each label
		dot := Dot(w, x)
		//                 fmt.Println(dot)
		scores = append(scores, dot)
		if dot > maxScore {
			maxScore = dot
			maxyId = j
		}
	}
	return cls.Labels.words[maxyId], maxScore
}

func LoadClassifier(fname string) Classifier {
	model_f, err := os.OpenFile(fname, os.O_RDONLY, 0644)
	if err != nil {
		panic("Failed to load model")
	}
	var (
		cls Classifier
	)
	cls.Labels = NewVocab()
	cls.Features = NewVocab()
	cls.weight = [][]float64{}
	reader := bufio.NewReaderSize(model_f, 4096*32)
	for {
		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		text := string(line)
		elems := strings.Split(text, "\t")
		label := elems[0]
		id := elems[1]
		w, _ := strconv.ParseFloat(elems[2], 64)
		labelid := cls.Labels.getId(label, true)
		ftid := cls.Features.getId(id, true)

		for len(cls.weight) < labelid+1 {
			cls.weight = append(cls.weight, []float64{})
		}
		for len(cls.weight[labelid]) < ftid+1 {
			cls.weight[labelid] = append(cls.weight[labelid], 0.)
		}
		cls.weight[labelid][ftid] = w

	}
	return cls
}
