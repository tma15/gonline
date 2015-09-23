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

type Feature struct {
	Id   int
	Val  float64
	Name string
}

func NewFeature(id int, val float64, name string) Feature {
	f := Feature{
		Id:   id,
		Val:  val,
		Name: name,
	}
	return f
}

type Param Feature

func (this *Dict) AddElem(elem string) {
	id := len(this.Id2elem)
	this.Elem2id[elem] = id
	this.Id2elem = append(this.Id2elem, elem)
}

type LearnerInterface interface {
	Name() string
	conv(*string, *map[string]float64) (int, *[]Feature)
	Fit(*[]map[string]float64, *[]string)
	Save(string)
	GetParam() *[][]float64
	GetParams() *[][][]float64
	//     GetNonZeroParams() (*[][][]Param, *[]string)
	GetNonZeroParams() *[][][]Param
	GetDics() (*Dict, *Dict)
	SetParam(*[][]float64)
	SetParams(*[][][]float64)
	SetDics(*Dict, *Dict)
}

type Learner struct {
	Weight    [][]float64
	FtDict    Dict
	LabelDict Dict
}

func (this *Learner) conv(label *string, x *map[string]float64) (int, *[]Feature) {
	num_feat := len(*x)
	features := make([]Feature, num_feat, num_feat)
	i := 0
	for name, val := range *x {
		if !this.FtDict.HasElem(name) {
			this.FtDict.AddElem(name)
		}
		id := this.FtDict.Elem2id[name]
		f := NewFeature(id, val, name)
		features[i] = f
		i++
	}
	if !this.LabelDict.HasElem(*label) {
		this.LabelDict.AddElem(*label)
	}
	tid := this.LabelDict.Elem2id[*label]
	return tid, &features
}

func (this *Learner) Name() string {
	return "Learner"
}

func (this *Learner) Fit(*[]map[string]float64, *[]int) {
	/* does not implement */
}

func (this *Learner) Save(fname string) {
	fp, err := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic(err)
	}
	writer := bufio.NewWriterSize(fp, 4096*32)
	labelsize := len(this.LabelDict.Id2elem)
	writer.WriteString(fmt.Sprintf("labelsize\t%d\n", labelsize))
	for i := 0; i < labelsize; i++ {
		label := this.LabelDict.Id2elem[i]
		writer.WriteString(fmt.Sprintf("%s\n", label))
	}
	for labelid := 0; labelid < labelsize; labelid++ {
		w := this.Weight[labelid]
		featuresize := len(w)
		writer.WriteString(fmt.Sprintf("featuresize\t%d\n", featuresize))
		for ftid := 0; ftid < len(w); ftid++ {
			ft := this.FtDict.Id2elem[ftid]
			writer.WriteString(fmt.Sprintf("%s %f\n", ft, w[ftid]))
		}
	}
	writer.Flush()
}

func (this *Learner) GetParam() *[][]float64 {
	return &this.Weight
}

func (this *Learner) GetParams() *[][][]float64 {
	params := [][][]float64{
		this.Weight,
	}
	return &params
}

func (this *Learner) GetDics() (*Dict, *Dict) {
	return &this.FtDict, &this.LabelDict
}

func (this *Learner) SetParam(w *[][]float64) {
	this.Weight = *w
}

func (this *Learner) SetParams(params *[][][]float64) {
	this.Weight = (*params)[0]
}

func (this *Learner) SetDics(ftdict, labeldict *Dict) {
	this.FtDict = *ftdict
	this.LabelDict = *labeldict
}

func (this *Learner) GetNonZeroParams() *[][][]Param {
	//     labels := make([]string, 0, 10)
	params := make([][][]Param, 0, 1)
	params = append(params, make([][]Param, 0, 10))
	for yid := 0; yid < len(this.Weight); yid++ {
		params[0] = append(params[0], make([]Param, 0, 1000))
		//         label := this.LabelDict.Id2elem[yid]
		//         labels = append(labels, label)
		for ftid := 0; ftid < len(this.Weight[ftid]); ftid++ {
			if this.Weight[yid][ftid] != 0. {
				ft := this.FtDict.Id2elem[ftid]
				param := Param{
					Id:   ftid,
					Name: ft,
					Val:  this.Weight[yid][ftid],
				}
				params[0][yid] = append(params[0][yid], param)
			}
		}
	}
	//     return &params, &labels
	return &params
}

type Perceptron struct {
	*Learner
}

func NewPerceptron() *Perceptron {
	p := Perceptron{&Learner{}}
	p.Weight = make([][]float64, 0, 1000)
	p.FtDict = NewDict()
	p.LabelDict = NewDict()
	return &p
}

func (this *Perceptron) Name() string {
	return "Perceptron"
}

func (this *Perceptron) Fit(x *[]map[string]float64, y *[]string) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]
		tid, features := this.conv(&yi, &xi)
		if len(this.Weight) <= tid {
			for k := len(this.Weight); k <= tid; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)
		for yid := 0; yid < len(this.Weight); yid++ {
			w := &this.Weight[yid]
			dot := 0.
			for _, feature := range *features {
				/* expand feature size */
				if len(*w) <= feature.Id {
					for k := len(*w); k <= feature.Id; k++ {
						*w = append(*w, 0.)
					}
				}

				if feature.Id >= len(*w) {
					continue
				}
				dot += (*w)[feature.Id] * feature.Val
			}

			if max < dot {
				max = dot
				argmax = yid
			}
		}

		if argmax != tid {
			for _, feature := range *features {
				this.Weight[tid][feature.Id] += feature.Val
				this.Weight[argmax][feature.Id] -= feature.Val
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
	pa.FtDict = NewDict()
	pa.LabelDict = NewDict()
	return &pa
}

func (this *PA) Name() string {
	return "PA"
}

func (this *PA) Fit(x *[]map[string]float64, y *[]string) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]
		tid, features := this.conv(&yi, &xi)

		/* expand label size */
		if len(this.Weight) <= tid {
			for k := len(this.Weight); k <= tid; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)

		for yid := 0; yid < len(this.Weight); yid++ {
			w := &this.Weight[yid]
			dot := 0.
			for _, feature := range *features {
				/* expand feature size */
				if len(*w) <= feature.Id {
					for k := len(*w); k <= feature.Id; k++ {
						*w = append(*w, 0.)
					}
				}

				if feature.Id >= len(*w) {
					continue
				}
				dot += (*w)[feature.Id] * feature.Val
			}

			if max < dot {
				max = dot
				argmax = yid
			}
		}

		if argmax != tid {
			norm := 0.
			for _, feature := range *features {
				norm += feature.Val * feature.Val
			}
			tau := this.Tau(max, norm, this.C)
			for _, feature := range *features {
				this.Weight[tid][feature.Id] += tau * feature.Val
				this.Weight[argmax][feature.Id] -= tau * feature.Val
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
	Diag [][]float64
}

func NewCW(eta float64) *CW {
	cdf := Normal_CDF(0.0, 1.0)
	cw := CW{&Learner{}, 1., cdf(eta), make([][]float64, 0, 100)}
	cw.FtDict = NewDict()
	cw.LabelDict = NewDict()

	return &cw
}

func (this *CW) Name() string {
	return "CW"
}

func (this *CW) GetParams() *[][][]float64 {
	params := [][][]float64{
		this.Weight,
		this.Diag,
	}
	return &params
}

func (this *CW) SetParams(params *[][][]float64) {
	this.Weight = (*params)[0]
	this.Diag = (*params)[1]
}

func (this *CW) Fit(x *[]map[string]float64, y *[]string) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]
		tid, features := this.conv(&yi, &xi)

		/* expand label size */
		if len(this.Weight) <= tid {
			for k := len(this.Weight); k <= tid; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}

		if len(this.Diag) <= tid {
			for k := len(this.Diag); k <= tid; k++ {
				this.Diag = append(this.Diag, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)
		margins := make([]float64, len(this.Weight), len(this.Weight))

		for yid := 0; yid < len(this.Weight); yid++ {
			w := &this.Weight[yid]
			d := &this.Diag[yid]
			dot := 0.
			for _, feature := range *features {
				/* expand feature size */
				if len(*w) <= feature.Id {
					for k := len(*w); k <= feature.Id; k++ {
						*w = append(*w, 0.)
					}
				}
				if len(*d) <= feature.Id {
					for k := len(*d); k <= feature.Id; k++ {
						*d = append((*d), 1.*this.a)
					}
				}

				if feature.Id >= len(*w) {
					continue
				}
				dot += (*w)[feature.Id] * feature.Val
			}
			if max < dot {
				max = dot
				argmax = yid
			}
			margins[yid] = dot
		}
		if argmax == -1 {
			fmt.Println(max, argmax)
			os.Exit(1)
		}

		/* all constrants update where k=inf */
		for yid := 0; yid < len(this.Weight); yid++ {
			sign := 0.
			if yid == tid {
				sign = 1.
			} else {
				sign = -1.
			}
			M := sign * margins[yid]
			_Diag := &this.Diag[yid]
			V := calcConfidence(_Diag, features)
			_n := 1. + 2.*this.phi*M
			sqrt := math.Sqrt(math.Pow(_n, 2) - 8.*this.phi*(M-this.phi*V))
			gamma := (-1.*_n + sqrt) / (4. * this.phi * V)
			alpha := Max(0., gamma)
			beta := 2. * alpha * this.phi / (1. + 2.*alpha*this.phi*V)
			for _, feature := range *features {
				this.Weight[yid][feature.Id] += sign * alpha * (*_Diag)[feature.Id] * feature.Val
				(*_Diag)[feature.Id] -= (*_Diag)[feature.Id] * feature.Val * beta * feature.Val * (*_Diag)[feature.Id]
			}
		}
	}
}

/*
	- http://webee.technion.ac.il/people/koby/publications/arow_nips09.pdf
	- http://web.eecs.umich.edu/~kulesza/pubs/arow_mlj13.pdf
*/
type Arow struct {
	*Learner
	a     float64 /* initial variance parameter */
	gamma float64 /* regularization parameter */
	Diag  [][]float64
}

func NewArow(g float64) *Arow {
	arow := Arow{
		Learner: &Learner{},
		a:       1.,
		gamma:   g,
		Diag:    make([][]float64, 0, 100),
	}
	arow.FtDict = NewDict()
	arow.LabelDict = NewDict()
	return &arow
}

func (this *Arow) Name() string {
	return "AROW"
}

func (this *Arow) GetParams() *[][][]float64 {
	params := [][][]float64{
		this.Weight,
		this.Diag,
	}
	return &params
}

func (this *Arow) GetNonZeroParams() *[][][]Param {
	params := make([][][]Param, 0, 1)
	params = append(params, make([][]Param, 0, 10))
	params = append(params, make([][]Param, 0, 10))

	for yid := 0; yid < len(this.Weight); yid++ {
		params[0] = append(params[0], make([]Param, 0, 1000))
		params[1] = append(params[1], make([]Param, 0, 1000))
		for ftid := 0; ftid < len(this.Weight[yid]); ftid++ {
			if this.Weight[yid][ftid] != 0. {
				ft := this.FtDict.Id2elem[ftid]
				param := Param{
					Id:   ftid,
					Name: ft,
					Val:  this.Weight[yid][ftid],
				}
				params[0][yid] = append(params[0][yid], param)
			}
		}
		for ftid := 0; ftid < len(this.Diag[yid]); ftid++ {
			if this.Diag[yid][ftid] != 0. {
				ft := this.FtDict.Id2elem[ftid]
				param := Param{
					Id:   ftid,
					Name: ft,
					Val:  this.Diag[yid][ftid],
				}
				params[1][yid] = append(params[1][yid], param)
			}
		}
	}
	return &params
}

func (this *Arow) SetParams(params *[][][]float64) {
	this.Weight = (*params)[0]
	this.Diag = (*params)[1]
}

func (this *Arow) Fit(x *[]map[string]float64, y *[]string) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]
		tid, features := this.conv(&yi, &xi)

		/* expand label size */
		if len(this.Weight) <= tid {
			for k := len(this.Weight); k <= tid; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}
		if len(this.Diag) <= tid {
			for k := len(this.Diag); k <= tid; k++ {
				this.Diag = append(this.Diag, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)
		margins := make([]float64, len(this.Weight), len(this.Weight))
		for yid := 0; yid < len(this.Weight); yid++ {
			w := &this.Weight[yid]
			d := &this.Diag[yid]
			dot := 0.
			for _, feature := range *features {
				/* expand feature size */
				if len(*w) <= feature.Id {
					for k := len(*w); k <= feature.Id; k++ {
						*w = append(*w, 0.)
					}
				}
				if len(*d) <= feature.Id {
					for k := len(*d); k <= feature.Id; k++ {
						*d = append(*d, 1.*this.a)
					}
				}

				if feature.Id >= len(*w) {
					continue
				}
				dot += (*w)[feature.Id] * feature.Val
			}
			if max < dot {
				max = dot
				argmax = yid
			}
			margins[yid] = dot
		}
		if argmax == -1 {
			fmt.Println(features)
			fmt.Println(margins)
			os.Exit(1)
		}

		if margins[tid] < 1.+margins[argmax] {
			/* The full version of AROW algorithm */
			sign := 0.
			for yid := 0; yid < len(this.Weight); yid++ {
				if yid == tid {
					sign = 1.
				} else {
					sign = -1.
				}
				/* update parameters */
				_Diag := &this.Diag[yid]
				V := calcConfidence(_Diag, features)
				beta := 1. / (V + this.gamma)
				alpha := Max(0., 1.-sign*margins[yid]) * beta
				if alpha == 0. {
					continue
				}
				for _, feature := range *features {
					this.Weight[yid][feature.Id] += sign * alpha * (*_Diag)[feature.Id] * feature.Val
					(*_Diag)[feature.Id] -= beta * (*_Diag)[feature.Id] * feature.Val * feature.Val * (*_Diag)[feature.Id]
				}
			}
		}
	}
	fmt.Println("FITFIN", len(this.Weight), len(this.Diag))
}

/* Cumulative Distribution Function for the Normal distribution */
func Normal_CDF(mu, sigma float64) func(x float64) float64 {
	return func(x float64) float64 {
		return (0.5 * (1 - math.Erf((mu-x)/(sigma*math.Sqrt2))))
	}
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

func calcConfidence(Diag *[]float64, features *[]Feature) float64 {
	V := 0.
	for _, feature := range *features {
		V += feature.Val * feature.Val * (*Diag)[feature.Id]
	}
	return V
}
