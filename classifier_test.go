package gonline

import (
	"fmt"
	"testing"
)

func TestLoadModel(t *testing.T) {
	cls := LoadClassifier("./example.model")
	if len(cls.Weight) != 2 {
		t.Error(fmt.Sprintf("Invalid size of Weight:%d want:2", len(cls.Weight)))
	}
	if len(cls.FtDict.Id2elem) != 6 {
		t.Error(fmt.Sprintf("Invalid size of features:%d want:6", len(cls.FtDict.Elem2id)))
	}
	if len(cls.LabelDict.Id2elem) != 2 {
		t.Error(fmt.Sprintf("Invalid size of labels:%d want:2", len(cls.LabelDict.Elem2id)))
	}

	if cls.LabelDict.Elem2id["食べ物"] != 0 {
		t.Error(fmt.Sprintf("Invalid label id:%d want:0", cls.LabelDict.Elem2id["食べ物"]))
	}
	if cls.LabelDict.Elem2id["スポーツ"] != 1 {
		t.Error(fmt.Sprintf("Invalid label id:%d want:1", cls.LabelDict.Elem2id["スポーツ"]))
	}

	if cls.FtDict.Elem2id["焼き肉"] != 0 {
		t.Error(fmt.Sprintf("Invalid feature id:%d want:0", cls.FtDict.Elem2id["焼き肉"]))
	}
	if cls.FtDict.Elem2id["唐揚げ"] != 1 {
		t.Error(fmt.Sprintf("Invalid feature id:%d want:1", cls.FtDict.Elem2id["唐揚げ"]))
	}
	if cls.FtDict.Elem2id["ドリンク"] != 2 {
		t.Error(fmt.Sprintf("Invalid feature id:%d want:2", cls.FtDict.Elem2id["ドリンク"]))
	}
	if cls.FtDict.Elem2id["サッカー"] != 3 {
		t.Error(fmt.Sprintf("Invalid feature id:%d want:3", cls.FtDict.Elem2id["サッカー"]))
	}
	if cls.FtDict.Elem2id["野球"] != 4 {
		t.Error(fmt.Sprintf("Invalid feature id:%d want:4", cls.FtDict.Elem2id["野球"]))
	}
	if cls.FtDict.Elem2id["テニス"] != 5 {
		t.Error(fmt.Sprintf("Invalid feature id:%d want:5", cls.FtDict.Elem2id["テニス"]))
	}

	y := "食べ物"
	k := "焼き肉"
	yid := cls.LabelDict.Elem2id[y]
	kid := cls.FtDict.Elem2id[k]
	if cls.Weight[yid][kid] != 0.004634 {
		t.Error("failed to load weight of feature ``焼き肉'' in label ``食べ物''")
	}

	y = "スポーツ"
	k = "テニス"
	yid = cls.LabelDict.Elem2id[y]
	kid = cls.FtDict.Elem2id[k]
	if cls.Weight[yid][kid] != 0.059096 {
		t.Error(fmt.Sprintf("failed to load weight of feature ``テニス'' in label ``スポーツ'' :%f", cls.Weight[yid][kid]))
	}

}

func TestPredict(t *testing.T) {
	cls := LoadClassifier("./example.model")
	x := map[string]float64{
		"サッカー": 1.,
		"野球":   2.,
		"テニス":  1.,
		"ドリンク": 1.,
	}
	if cls.Predict(&x) != 1 {
		t.Error("Invalid predicted labelid")
	}
}
