package gonline

import (
	"fmt"
	"testing"
)

func TestDict(t *testing.T) {
	dict := NewDict()

	if dict.HasElem("焼き肉") == true {
		t.Error("焼き肉 exists in Dict")
	}

	if !dict.HasElem("焼き肉") {
		dict.AddElem("焼き肉")
	}
	if dict.HasElem("焼き肉") == false {
		t.Error("焼き肉 doesn't exist in Dict")
	}

	if len(dict.Id2elem) != 1 {
		t.Error(fmt.Sprintf("Invalid number of elements in dict:%d want:1", len(dict.Id2elem)))
	}
}

func TestPerceptron(t *testing.T) {
	leaner := NewPerceptron()
	x := []map[string]float64{
		map[string]float64{
			"焼き肉": 1.,
			"ピザ":  1.,
		},
	}
	y := []string{"食べ物"}
	leaner.Fit(&x, &y)
	dim_w := len((*leaner).Weight)
	if dim_w != 1 {
		t.Error(fmt.Sprintf("Invalid size of dimension of weight:%d want:1", dim_w))
	}
	dim_w_0 := len((*leaner).Weight[0])
	if dim_w_0 != 2 {
		t.Error(fmt.Sprintf("Invalid size of dimension of weight:%d want:2", dim_w_0))
	}

	x = []map[string]float64{
		map[string]float64{
			"サッカー": 1.,
			"野球":   1.,
		},
	}
	y = []string{"スポーツ"}
	leaner.Fit(&x, &y)

	dim_w = len((*leaner).Weight)
	if dim_w != 2 {
		t.Error(fmt.Sprintf("Invalid size of dimension of weight:%d want:2", dim_w))
	}
	dim_w_0 = len((*leaner).Weight[0])
	if dim_w_0 != 4 {
		t.Error(fmt.Sprintf("Invalid size of dimension of weight:%d want:4", dim_w_0))
	}

	ft := "サッカー"
	ftid := (*leaner).FtDict.Elem2id[ft]
	w := (*leaner).Weight[1][ftid]
	if w != 1. {
		t.Error(fmt.Sprintf("Invalid weight:%f want:1.", w))
	}
	w = (*leaner).Weight[0][ftid]
	if w != -1. {
		t.Error(fmt.Sprintf("Invalid weight:%f want:-1.", w))
	}

	ft = "野球"
	ftid = (*leaner).FtDict.Elem2id[ft]
	w = (*leaner).Weight[1][ftid]
	if w != 1. {
		t.Error(fmt.Sprintf("Invalid weight:%f want:1.", w))
	}
	w = (*leaner).Weight[0][ftid]
	if w != -1. {
		t.Error(fmt.Sprintf("Invalid weight:%f want:-1.", w))
	}
}
