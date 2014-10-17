package main

import (
	"bufio"
	"io"
	//         "encoding/json"
	"fmt"
	//         "io/ioutil"
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
		//                 fmt.Println(fv[1])
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
//                 fmt.Println(scanner.Text())
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
//         fmt.Println("<x>")
//         for i, x := range X{
//             fmt.Println(i, x)
//         }
//         fmt.Println("</x>")
	return X, y

}

func SaveModel(p Perceptron, fname string) {
	model_f, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("Failed to dump model")
	}

	writer := bufio.NewWriterSize(model_f, 4096*32)
	writer.WriteString("DEFAULTLABEL\n")
	writer.WriteString(fmt.Sprintf("%s\n", p.labelDefault))
	writer.WriteString("ETA\n")
	writer.WriteString(fmt.Sprintf("%f\n", p.eta))
	writer.WriteString("WEIGHT\n")
	for y, weight := range p.weight {
		for ft, w := range weight {
			writer.WriteString(fmt.Sprintf("%s\t%s\t%f\n", y, ft, w))
		}
	}
	writer.Flush()
}

func LoadModel(fname string) Perceptron {
	model_f, err := os.OpenFile(fname, os.O_RDONLY, 0644)
	if err != nil {
		panic("Failed to dump model")
	}
	weight := Weight{}
	var eta float64
	var labelDefault string
	reader := bufio.NewReaderSize(model_f, 4096)
	a := ""
	for {
		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		text := string(line)
		if text == "WEIGHT" {
			a = "w"
			continue
		} 
                if text == "ETA" {
			a = "e"
			continue
		} 
                if text == "DEFAULTLABEL" {
                    a = "d"
                    continue
                }

		if a == "w" {
			elems := strings.Split(text, "\t")
			label := elems[0]
			id := elems[1]
			w, _ := strconv.ParseFloat(elems[2], 64)
			_, ok := weight[label]
			if ok {
				//                                 fmt.Println(label, id, w)
				weight[label][id] = w
			} else {
				weight[label] = map[string]float64{}
			}
		} 
                if a == "e" {
			eta, _ = strconv.ParseFloat(text, 64)
		} 
                if a == "d" {
                    labelDefault = text
                }

	}
	p := Perceptron{weight, eta, labelDefault, 0}
	return p
}
