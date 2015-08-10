package gonline

import (
	"fmt"
	"testing"
)

func TestLoadData(t *testing.T) {
	x, y := LoadData("./example.txt")

	num_data_x := len(*x)
	if num_data_x != 2 {
		t.Error(fmt.Sprintf("Invalid number of datum:%d want:2", num_data_x))
	}
	num_data_y := len(*y)
	if num_data_y != 2 {
		t.Error(fmt.Sprintf("Invalid number of datum:%d want:2", num_data_y))
	}
	if num_data_y != num_data_x {
		t.Error(fmt.Sprintf("number of x and that of y is not the same: x:%d y:%d", num_data_x, num_data_y))
	}

	if (*y)[0] != "スポーツ" {
		t.Error(fmt.Sprintf("Invalid label:%s want:スポーツ", (*y)[0]))
	}
	k0 := []string{"サッカー", "野球", "テニス", "ドリンク"}
	v0 := []float64{1., 2., 1., 1.}
	x0 := (*x)[0]
	n := len(k0)
	for i := 0; i < n; i++ {
		if _, ok := x0[k0[i]]; !ok {
			t.Error(fmt.Sprintf("Failed to read key:%s", k0[i]))
		}
		if x0[k0[i]] != v0[i] {
			t.Error(fmt.Sprintf("Invalid value:%f want:%f", x0[k0[i]], v0[i]))
		}
	}

	if (*y)[1] != "食べ物" {
		t.Error(fmt.Sprintf("Invalid label:%s want:食べ物", (*y)[1]))
	}
	k1 := []string{"焼き肉", "唐揚げ", "ドリンク"}
	v1 := []float64{1., 1., 1.}
	x1 := (*x)[1]
	n = len(k1)
	for i := 0; i < n; i++ {
		if _, ok := x1[k1[i]]; !ok {
			t.Error(fmt.Sprintf("Failed to read key:%s", k1[i]))
		}
		if x1[k1[i]] != v1[i] {
			t.Error(fmt.Sprintf("Invalid value:%f want:%f", x1[k1[i]], v1[i]))
		}
	}
}

func TestShuffleData(t *testing.T) {
	x, y := LoadData("./example.txt")
	ShuffleData(x, y)

	num_data_x := len(*x)
	if num_data_x != 2 {
		t.Error(fmt.Sprintf("Invalid number of datum:%d want:2", num_data_x))
	}
	num_data_y := len(*y)
	if num_data_y != 2 {
		t.Error(fmt.Sprintf("Invalid number of datum:%d want:2", num_data_y))
	}
	if num_data_y != num_data_x {
		t.Error(fmt.Sprintf("number of x and that of y is not the same: x:%d y:%d", num_data_x, num_data_y))
	}
}
