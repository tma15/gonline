package gonline

import (
	"bufio"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func LoadFromStdin() ([]map[string]float64, []string) {
	scanner := bufio.NewScanner(os.Stdin)
	X := []map[string]float64{}
	y := []string{}
	for scanner.Scan() {
		fv := strings.SplitN(scanner.Text(), " ", 2)
		y_i := fv[0]
		x := map[string]float64{}
		for _, k := range strings.Split(strings.Trim(fv[1], " "), " ") {
			i := strings.Split(k, ":")
			i64, _ := strconv.ParseFloat(i[1], 64)
			x[i[0]] = i64
		}
		X = append(X, x)
		y = append(y, y_i)
	}
	return X, y
}

func ShuffleData(x *[]map[string]float64, y *[]string) {
	rand.Seed(time.Now().UnixNano())
	numdata := len(*x)
	idx := make([]int, numdata, numdata)
	for i := 0; i < numdata; i++ {
		idx[i] = i
	}
	for i := range idx {
		j := rand.Intn(i + 1)
		(*x)[i], (*x)[j] = (*x)[j], (*x)[i]
		(*y)[i], (*y)[j] = (*y)[j], (*y)[i]
	}
}

func LoadData(fname string) (*[]map[string]float64, *[]string) {
	x := make([]map[string]float64, 0, 100000)
	y := make([]string, 0, 100000)
	fp, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	reader := bufio.NewReaderSize(fp, 4096*64)
	lid := 0
	for {
		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		if strings.HasPrefix(string(line), "#") {
			continue /* ignore comment */
		}
		fv := strings.SplitN(string(line), " ", 2)
		if len(fv) != 2 {
			fmt.Println(fmt.Sprintf("line:%d has no features. This line is ignored.", lid))
			continue
		}
		label_i := fv[0]

		features := strings.Split(strings.Trim(fv[1], " "), " ")
		x_i := make(map[string]float64)

		for _, k := range features {
			sp := strings.Split(k, ":")
			if len(sp) != 2 {
				continue
			}
			name := sp[0]
			v, err := strconv.ParseFloat(sp[1], 64)
			if err != nil {
				panic(err)
			}
			x_i[name] = v
		}
		x = append(x, x_i)
		y = append(y, label_i)
		lid++
	}
	return &x, &y
}
