package gonline

import (
	"bufio"
	"fmt"
	"math"
	"os"
)

type Dict struct {
	Id2elem []string
	Elem2id map[string]int
}

func NewDict() Dict {
	d := Dict{Id2elem: make([]string, 0, 10000),
		Elem2id: make(map[string]int)}
	return d
}

func (this *Dict) HasElem(elem string) bool {
	if _, ok := this.Elem2id[elem]; !ok {
		return false
	} else {
		return true
	}
}

func (this *Dict) AddElem(elem string) {
	id := len(this.Id2elem)
	this.Elem2id[elem] = id
	this.Id2elem = append(this.Id2elem, elem)
}

func conv2fv(ftdict *Dict, xmap map[string]float64) *[]Feature {
	x := make([]Feature, len(xmap), len(xmap))
	id := 0
	for name, val := range xmap {
		x[id] = NewFeature(id, val, name)
		id += 1
	}
	return &x
}

type Feature struct {
	Id    int
	Val   float64
	Label string
}

func NewFeature(id int, val float64, label string) Feature {
	return Feature{Id: id, Val: val, Label: label}
}

type LearnerInterface interface {
	Name() string
	Fit(*[][]Feature, *[]int)
	Save(string, *Dict, *Dict)
	GetParam() *[][]float64
	SetParam(*[][]float64)
}

type Learner struct {
	Weight [][]float64
}

func (this *Learner) Name() string {
	return "Learner"
}

func (this *Learner) Fit(*[][]Feature, *[]int) {
	/* does not implement */
}

func (this *Learner) Save(fname string, ftdict *Dict, labeldict *Dict) {
	fp, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic(err)
	}
	writer := bufio.NewWriterSize(fp, 4096*32)
	labelsize := len((*labeldict).Id2elem)
	ftsize := len((*ftdict).Id2elem)
	writer.WriteString(fmt.Sprintf("labelsize\t%d\n", labelsize))
	writer.WriteString(fmt.Sprintf("featuresize\t%d\n", ftsize))
	for i := 0; i < labelsize; i++ {
		label := (*labeldict).Id2elem[i]
		writer.WriteString(fmt.Sprintf("%s\n", label))
	}
	for i := 0; i < ftsize; i++ {
		ft := (*ftdict).Id2elem[i]
		writer.WriteString(fmt.Sprintf("%s\n", ft))
	}
	for labelid := 0; labelid < labelsize; labelid++ {
		w := this.Weight[labelid]
		for ftid := 0; ftid < ftsize; ftid++ {
			writer.WriteString(fmt.Sprintf("%f\n", w[ftid]))
		}
	}
	writer.Flush()
}

func (this *Learner) GetParam() *[][]float64 {
	return &this.Weight
}

func (this *Learner) SetParam(w *[][]float64) {
	this.Weight = *w
}

type Perceptron struct {
	*Learner
}

func NewPerceptron() *Perceptron {
	l := Perceptron{&Learner{}}
	l.Weight = make([][]float64, 0, 1000)
	return &l
}

func (this *Perceptron) Name() string {
	return "Perceptron"
}

func (this *Perceptron) Fit(x *[][]Feature, y *[]int) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]

		/* expand label size */
		if len(this.Weight) <= yi {
			for k := len(this.Weight); k <= yi; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)

		for labelid := 0; labelid < len(this.Weight); labelid++ {

			w := &this.Weight[labelid]
			dot := 0.
			for _, ft := range xi {

				/* expand feature size */
				if len(*w) <= ft.Id {
					for k := len(*w); k <= ft.Id; k++ {
						*w = append(*w, 0.)
					}
				}

				dot += (*w)[ft.Id] * ft.Val
			}

			if max < dot {
				max = dot
				argmax = labelid
			}
		}

		if argmax != yi {
			for _, ft := range xi {
				this.Weight[yi][ft.Id] += ft.Val
				this.Weight[argmax][ft.Id] -= ft.Val
			}
		}
	}
}

/*
http://www.jmlr.org/papers/volume7/crammer06a/crammer06a.pdf
*/
type PA struct {
	*Learner
	C   float64 /* degree of aggressiveness */
	Tau func(float64, float64, float64) float64
}

func NewPA(mode string, C float64) *PA {
	var pa PA
	switch mode {
	case "":
		pa = PA{&Learner{}, C, tau} /* PA doesn't use C actually */
	case "I":
		pa = PA{&Learner{}, C, tauI}
	case "II":
		pa = PA{&Learner{}, C, tauII}
	default:
		os.Exit(1)
	}
	pa.Weight = make([][]float64, 0, 1000)
	return &pa
}

func (this *PA) Name() string {
	return "PA"
}

func (this *PA) Fit(x *[][]Feature, y *[]int) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]

		/* expand label size */
		if len(this.Weight) <= yi {
			for k := len(this.Weight); k <= yi; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)

		for labelid := 0; labelid < len(this.Weight); labelid++ {
			w := &this.Weight[labelid]
			dot := 0.
			for _, ft := range xi {

				/* expand feature size */
				if len(*w) <= ft.Id {
					for k := len(*w); k <= ft.Id; k++ {
						*w = append(*w, 0.)
					}
				}

				dot += (*w)[ft.Id] * ft.Val
			}

			if max < dot {
				max = dot
				argmax = labelid
			}
		}

		if argmax != yi {
			norm := 0.
			for _, ft := range xi {
				norm += ft.Val * ft.Val
			}
			for _, ft := range xi {
				tau := this.Tau(max, norm, this.C)
				this.Weight[yi][ft.Id] += tau * ft.Val
				this.Weight[argmax][ft.Id] -= tau * ft.Val
			}
		}
	}
}

func tau(max, norm, C float64) float64 {
	return (1 + max) / norm
}

func tauI(max, norm, C float64) float64 {
	return Min(C, (1+max)/norm)
}

func tauII(max, norm, C float64) float64 {
	return (1 + max) / (norm + 1/(2.*C))
}

/*
- http://www.cs.jhu.edu/~mdredze/publications/icml_variance.pdf
- http://www.aclweb.org/anthology/D09-1052
- http://www.jmlr.org/papers/volume13/crammer12a/crammer12a.pdf
*/
type CW struct {
	*Learner
	a    float64 /* initial variance parameter */
	phi  float64
	diag [][]float64
}

func NewCW(eta float64) *CW {
	cdf := Normal_CDF(0.0, 1.0)
	cw := CW{&Learner{}, 1., cdf(eta), make([][]float64, 0, 100)}
	return &cw
}

func (this *CW) Name() string {
	return "CW"
}

func (this *CW) Fit(x *[][]Feature, y *[]int) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]

		/* expand label size */
		if len(this.Weight) <= yi {
			for k := len(this.Weight); k <= yi; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}
		if len(this.diag) <= yi {
			for k := len(this.diag); k <= yi; k++ {
				this.diag = append(this.diag, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)
		margins := make([]float64, len(this.Weight), len(this.Weight))

		for labelid := 0; labelid < len(this.Weight); labelid++ {
			w := &this.Weight[labelid]
			d := &this.diag[labelid]
			dot := 0.
			for _, ft := range xi {

				/* expand feature size */
				if len(*w) <= ft.Id {
					for k := len(*w); k <= ft.Id; k++ {
						*w = append(*w, 0.)
					}
				}
				if len(*d) <= ft.Id {
					for k := len(*d); k <= ft.Id; k++ {
						*d = append((*d), 1.*this.a)
					}
				}

				dot += (*w)[ft.Id] * ft.Val
			}
			if max < dot {
				max = dot
				argmax = labelid
			}
			margins[labelid] = dot
		}
		if argmax == -1 {
			fmt.Println(max, argmax)
			os.Exit(1)
		}

		/* update parameters of true label */
		M := margins[yi]
		_diag := &this.diag[yi]
		V := 0.
		for _, ft := range xi {
			V += ft.Val * ft.Val * (*_diag)[ft.Id]
		}
		_n := 1. + 2.*this.phi*M
		sqrt := math.Sqrt(math.Pow(_n, 2) - 8.*this.phi*(M-this.phi*V))
		gamma := (-1.*_n + sqrt) / (4. * this.phi * V)
		alpha := Max(0., gamma)
		beta := 2. * alpha * this.phi / (1. + 2.*alpha*this.phi*V)
		for _, ft := range xi {
			this.Weight[yi][ft.Id] += alpha * (*_diag)[ft.Id] * ft.Val
			(*_diag)[ft.Id] -= (*_diag)[ft.Id] * ft.Val * beta * ft.Val * (*_diag)[ft.Id]
		}

		/* update parameters of predicted label */
		_diag = &this.diag[argmax]
		V = 0.
		for _, ft := range xi {
			V += ft.Val * ft.Val * (*_diag)[ft.Id]
		}
		_n = 1. + 2.*this.phi*M
		sqrt = math.Sqrt(math.Pow(_n, 2) - 8.*this.phi*(M-this.phi*V))
		gamma = (-1.*_n + sqrt) / (4. * this.phi * V)
		alpha = Max(0., gamma)
		beta = 2. * alpha * this.phi / (1. + 2.*alpha*this.phi*V)
		for _, ft := range xi {
			this.Weight[argmax][ft.Id] -= alpha * (*_diag)[ft.Id] * ft.Val
			(*_diag)[ft.Id] -= (*_diag)[ft.Id] * ft.Val * beta * ft.Val * (*_diag)[ft.Id]
		}
	}
}

/*
- http://webee.technion.ac.il/people/koby/publications/arow_nips09.pdf
*/
type Arow struct {
	*Learner
	a     float64 /* initial variance parameter */
	gamma float64 /* regularization parameter */
	diag  [][]float64
}

func NewArow(gamma float64) *Arow {
	arow := Arow{&Learner{}, 1., gamma, make([][]float64, 0, 100)}
	return &arow
}

func (this *Arow) Name() string {
	return "AROW"
}

func (this *Arow) Fit(x *[][]Feature, y *[]int) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]

		/* expand label size */
		if len(this.Weight) <= yi {
			for k := len(this.Weight); k <= yi; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}
		if len(this.diag) <= yi {
			for k := len(this.diag); k <= yi; k++ {
				this.diag = append(this.diag, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)
		margins := make([]float64, len(this.Weight), len(this.Weight))
		for labelid := 0; labelid < len(this.Weight); labelid++ {
			w := &this.Weight[labelid]
			d := &this.diag[labelid]
			dot := 0.
			for _, ft := range xi {

				/* expand feature size */
				if len(*w) <= ft.Id {
					for k := len(*w); k <= ft.Id; k++ {
						*w = append(*w, 0.)
					}
				}
				if len(*d) <= ft.Id {
					for k := len(*d); k <= ft.Id; k++ {
						*d = append(*d, 1.*this.a)
					}
				}

				dot += (*w)[ft.Id] * ft.Val
			}
			if max < dot {
				max = dot
				argmax = labelid
			}
			margins[labelid] = dot
		}
		if argmax == -1 {
			fmt.Println(max, argmax)
			os.Exit(1)
		}

		/* update parameters of true label */
		loss := Max(0., 1.+margins[argmax]-margins[yi])
		_diag := &this.diag[yi]
		V := 0.
		for _, ft := range xi {
			V += ft.Val * ft.Val * (*_diag)[ft.Id]
		}
		beta := 1. / (V + this.gamma)
		alpha := loss * beta
		for _, ft := range xi {
			this.Weight[yi][ft.Id] += alpha * (*_diag)[ft.Id] * ft.Val
			(*_diag)[ft.Id] -= (*_diag)[ft.Id] * ft.Val * beta * ft.Val * (*_diag)[ft.Id]
		}

		/* update parameters of predicted label */
		_diag = &this.diag[argmax]
		V = 0.
		for _, ft := range xi {
			V += ft.Val * ft.Val * (*_diag)[ft.Id]
		}
		beta = 1. / (V + this.gamma)
		alpha = loss * beta
		for _, ft := range xi {
			this.Weight[argmax][ft.Id] -= alpha * (*_diag)[ft.Id] * ft.Val
			(*_diag)[ft.Id] -= (*_diag)[ft.Id] * ft.Val * beta * ft.Val * (*_diag)[ft.Id]
		}
	}
}

/* Cumulative Distribution Function for the Normal distribution */
func Normal_CDF(mu, sigma float64) func(x float64) float64 {
	return func(x float64) float64 { return (0.5 * (1 - math.Erf((mu-x)/(sigma*math.Sqrt2)))) }
}

func Min(x, y float64) float64 {
	if x > y {
		return y
	} else {
		return x
	}
}

func Max(x, y float64) float64 {
	if x > y {
		return x
	} else {
		return y
	}
}
