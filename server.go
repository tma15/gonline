package gonline

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
)

type Data struct {
	X *[]map[string]float64 `json:"x"`
	Y *[]string             `json:"y"`
}

func (this *Data) GetBatch(start, end int) *Data {
	x := (*this.X)[start:end]
	y := (*this.Y)[start:end]
	batch := Data{
		X: &x,
		Y: &y,
	}
	return &batch
}

type Model struct {
	Algorightm string         `json:"a"`
	Id2Feature []string       `json:"id2f"`
	Feature2Id map[string]int `json:"f2id"`
	Params     [][][]float64  `json:"params"`
	Id2Label   []string       `json:"id2y"`
	Label2Id   map[string]int `json:"y2id"`
}

type LearnerServer struct {
	Learner LearnerInterface
	Host    string
	Port    string
}

func NewLearnerServer(host, port string) LearnerServer {
	var learner LearnerInterface
	learner = NewArow(1.)
	this := LearnerServer{
		Learner: learner,
		Port:    port,
		Host:    host,
	}
	return this
}

func (this *LearnerServer) Start() {
	serverMux := http.NewServeMux()
	serverMux.HandleFunc("/fit", this.fit)
	serverMux.HandleFunc("/average", this.getAveragedModel)
	err := http.ListenAndServe(fmt.Sprintf("%s:%s", this.Host, this.Port), serverMux)
	if err != nil {
		panic(err)
	}
}

func (this *LearnerServer) fit(w http.ResponseWriter, r *http.Request) {
	rb := bufio.NewReader(r.Body)
	request := ""
	for {
		s, err := rb.ReadString('\n')
		request += s
		if err == io.EOF {
			break
		}
	}
	var data Data
	err := json.Unmarshal([]byte(request), &data)
	if err != nil {
		panic(err)
	}
	this.Learner.Fit(data.X, data.Y)

	w.Header().Set("Content-Type:", "application/json")

	params := this.Learner.GetParams()
	ftdict, labeldict := this.Learner.GetDics()

	model := Model{
		Algorightm: this.Learner.Name(),
		Params:     *params,
		Feature2Id: ftdict.Elem2id,
		Id2Feature: ftdict.Id2elem,
		Label2Id:   labeldict.Elem2id,
		Id2Label:   labeldict.Id2elem,
	}
	json, err := json.Marshal(model)
	if err != nil {
		panic(err)
	}

	fmt.Fprintf(w, string(json))
}

func (this *LearnerServer) getAveragedModel(w http.ResponseWriter, r *http.Request) {
	rb := bufio.NewReader(r.Body)
	request := ""
	for {
		s, err := rb.ReadString('\n')
		request += s
		if err == io.EOF {
			break
		}
	}
	var model Model
	err := json.Unmarshal([]byte(request), &model)
	if err != nil {
		panic(err)
	}

	this.Learner.SetParams(&model.Params)

	ftdict := NewDict()
	ftdict.Id2elem = model.Id2Feature
	ftdict.Elem2id = model.Feature2Id

	labeldict := NewDict()
	labeldict.Id2elem = model.Id2Label
	labeldict.Elem2id = model.Label2Id

	this.Learner.SetDics(&ftdict, &labeldict)
}

type Client struct {
}

func NewClient() Client {
	this := Client{}
	return this
}

func (this *Client) SendData(host, port string, data *Data) *LearnerInterface {
	jsonIn, err := json.Marshal(*data)
	if err != nil {
		panic(err)
	}

	addr := fmt.Sprintf("http://%s:%s", host, port)
	resp, err := http.Post(fmt.Sprintf("%s/fit", addr),
		"application/json",
		bytes.NewBuffer(jsonIn))
	if err != nil {
		panic(err)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	var model Model
	err = json.Unmarshal(body, &model)
	if err != nil {
		panic(err)
	}

	var learner LearnerInterface
	switch model.Algorightm {
	case "AROW":
		learner = NewArow(1.)
	default:
		fmt.Println(model.Algorightm)
		os.Exit(1)
	}
	learner.SetParams(&model.Params)

	ftdict := NewDict()
	ftdict.Id2elem = model.Id2Feature
	ftdict.Elem2id = model.Feature2Id

	labeldict := NewDict()
	labeldict.Id2elem = model.Id2Label
	labeldict.Elem2id = model.Label2Id

	learner.SetDics(&ftdict, &labeldict)
	return &learner
}

func (this *Client) SendModel(host, port string, learner *LearnerInterface) {
	params := (*learner).GetParams()
	ftdict, labeldict := (*learner).GetDics()

	model := Model{
		Algorightm: (*learner).Name(),
		Params:     *params,
		Feature2Id: ftdict.Elem2id,
		Id2Feature: ftdict.Id2elem,
		Label2Id:   labeldict.Elem2id,
		Id2Label:   labeldict.Id2elem,
	}

	jsonIn, err := json.Marshal(model)
	if err != nil {
		panic(err)
	}

	addr := fmt.Sprintf("http://%s:%s", host, port)
	_, err = http.Post(fmt.Sprintf("%s/average", addr),
		"application/json",
		bytes.NewBuffer(jsonIn))
	if err != nil {
		panic(err)
	}
}
