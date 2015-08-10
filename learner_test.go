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
