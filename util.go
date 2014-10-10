package main

import (
	"bufio"
	"encoding/json"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

func LoadFromStdin() ([]map[string]float64, []string) {
	scanner := bufio.NewScanner(os.Stdin)
	X := []map[string]float64{}
	y := []string{}
	for scanner.Scan() {
		fv := strings.SplitN(scanner.Text(), " ", 2)
		y_i := fv[0]
		x := map[string]float64{}
		//fmt.Println(fv[1])
		for _, k := range strings.Split(strings.Trim(fv[1], " "), " ") {
			//fmt.Println(k)
			i := strings.Split(k, ":")
			i64, _ := strconv.ParseFloat(i[1], 64)
			x[i[0]] = i64
		}
		X = append(X, x)
		y = append(y, y_i)
	}
	return X, y
}

func LoadFromFile(fname string) ([]map[string]float64, []string) {
    fp, err := os.Open(fname)
    if err != nil {
        panic(err)
    }

    scanner := bufio.NewScanner(fp)
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

func SaveModel(weight map[string]map[string]float64, fname string) {
        model_f, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
        if err != nil {
                panic("Failed to dump model")
        }
        encoder := json.NewEncoder(model_f)
        encoder.Encode(weight)
}

func LoadModel(fname string) map[string]map[string]float64 {
        model_f, err := ioutil.ReadFile(fname)
        if err != nil {
                panic("Failed to dump model")
        }
        var f interface{}
        json.Unmarshal(model_f, &f)
        ff := f.(map[string]interface{})
        weight := map[string]map[string]float64{}
        for k, v := range ff {
            fv := v.(map[string]interface{})
            weight[k] = map[string]float64{}
            for i, j := range fv {
                weight[k][i] = j.(float64)
            }
        }
        return weight
}

