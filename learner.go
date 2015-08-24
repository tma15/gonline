package gonline

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
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

type LearnerInterface interface {
	Name() string
	updateDict(*[]string, *string)
	Fit(*[]map[string]float64, *[]string)
	Save(string)
	GetParam() *[][]float64
	GetParams() *[][][]float64
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

func (this *Learner) updateDict(features *[]string, label *string) {
	for _, name := range *features {
		if !this.FtDict.HasElem(name) {
			this.FtDict.AddElem(name)
		}
	}
	if !this.LabelDict.HasElem(*label) {
		this.LabelDict.AddElem(*label)
	}
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
	featuresize := len(this.FtDict.Id2elem)
	writer.WriteString(fmt.Sprintf("labelsize\t%d\n", labelsize))
	writer.WriteString(fmt.Sprintf("featuresize\t%d\n", featuresize))
	for i := 0; i < labelsize; i++ {
		label := this.LabelDict.Id2elem[i]
		writer.WriteString(fmt.Sprintf("%s\n", label))
	}
	for i := 0; i < featuresize; i++ {
		ft := this.FtDict.Id2elem[i]
		writer.WriteString(fmt.Sprintf("%s\n", ft))
	}
	for labelid := 0; labelid < labelsize; labelid++ {
		w := this.Weight[labelid]
		for ftid := 0; ftid < featuresize; ftid++ {
			writer.WriteString(fmt.Sprintf("%f\n", w[ftid]))
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
		features := GetSortedFeatures(&xi)
		this.updateDict(&features, &yi)

		tid := this.LabelDict.Elem2id[yi]
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
			for _, ft := range features {
				val := xi[ft]
				ftid := this.FtDict.Elem2id[ft]

				/* expand feature size */
				if len(*w) <= ftid {
					for k := len(*w); k <= ftid; k++ {
						*w = append(*w, 0.)
					}
				}

				dot += (*w)[ftid] * val
			}

			if max < dot {
				max = dot
				argmax = yid
			}
		}

		if argmax != tid {
			for _, ft := range features {
				val := xi[ft]
				ftid := this.FtDict.Elem2id[ft]
				this.Weight[tid][ftid] += val
				this.Weight[argmax][ftid] -= val
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
		features := GetSortedFeatures(&xi)
		this.updateDict(&features, &yi)
		tid := this.LabelDict.Elem2id[yi]

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
			for _, ft := range features {
				val := xi[ft]
				ftid := this.FtDict.Elem2id[ft]
				/* expand feature size */
				if len(*w) <= ftid {
					for k := len(*w); k <= ftid; k++ {
						*w = append(*w, 0.)
					}
				}
				dot += (*w)[ftid] * val
			}

			if max < dot {
				max = dot
				argmax = yid
			}
		}

		if argmax != tid {
			norm := 0.
			for _, ft := range features {
				val := xi[ft]
				norm += val * val
			}
			tau := this.Tau(max, norm, this.C)
			for _, ft := range features {
				val := xi[ft]
				ftid := this.FtDict.Elem2id[ft]
				this.Weight[tid][ftid] += tau * val
				this.Weight[argmax][ftid] -= tau * val
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
		this.diag,
	}
	return &params
}

func (this *CW) SetParams(params *[][][]float64) {
	this.Weight = (*params)[0]
	this.diag = (*params)[1]
}

func (this *CW) Fit(x *[]map[string]float64, y *[]string) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]
		features := GetSortedFeatures(&xi)
		this.updateDict(&features, &yi)

		tid := this.LabelDict.Elem2id[yi]

		/* expand label size */
		if len(this.Weight) <= tid {
			for k := len(this.Weight); k <= tid; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}

		if len(this.diag) <= tid {
			for k := len(this.diag); k <= tid; k++ {
				this.diag = append(this.diag, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)
		margins := make([]float64, len(this.Weight), len(this.Weight))

		for yid := 0; yid < len(this.Weight); yid++ {
			w := &this.Weight[yid]
			d := &this.diag[yid]
			dot := 0.
			for _, ft := range features {
				val := xi[ft]
				ftid := this.FtDict.Elem2id[ft]

				/* expand feature size */
				if len(*w) <= ftid {
					for k := len(*w); k <= ftid; k++ {
						*w = append(*w, 0.)
					}
				}
				if len(*d) <= ftid {
					for k := len(*d); k <= ftid; k++ {
						*d = append((*d), 1.*this.a)
					}
				}

				dot += (*w)[ftid] * val
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

		/* all constrants update */
		for yid := 0; yid < len(this.Weight); yid++ {
			sign := 0.
			if yid == tid {
				sign = 1.
			} else {
				sign = -1.
			}
			M := sign * margins[yid]
			_diag := &this.diag[yid]
			V := calcConfidence(_diag, &xi, &this.FtDict)
			_n := 1. + 2.*this.phi*M
			sqrt := math.Sqrt(math.Pow(_n, 2) - 8.*this.phi*(M-this.phi*V))
			gamma := (-1.*_n + sqrt) / (4. * this.phi * V)
			alpha := Max(0., gamma)
			beta := 2. * alpha * this.phi / (1. + 2.*alpha*this.phi*V)
			for _, ft := range features {
				val := xi[ft]
				ftid := this.FtDict.Elem2id[ft]
				this.Weight[yid][ftid] += sign * alpha * (*_diag)[ftid] * val
				(*_diag)[ftid] -= (*_diag)[ftid] * val * beta * val * (*_diag)[ftid]
			}
		}

		//         /* single constraint update */
		/* update parameters of true label */
		//         M := margins[tid]
		//         _diag := &this.diag[tid]
		//         Vtrue := calcConfidence(_diag, &xi, &this.FtDict)
		//         _n := 1. + 2.*this.phi*M
		//         sqrt := math.Sqrt(math.Pow(_n, 2) - 8.*this.phi*(M-this.phi*Vtrue))
		//         gamma := (-1.*_n + sqrt) / (4. * this.phi * Vtrue)
		//         alpha := Max(0., gamma)
		//         beta := 2. * alpha * this.phi / (1. + 2.*alpha*this.phi*Vtrue)
		//         for _, ft := range features {
		//             val := xi[ft]
		//             ftid := this.FtDict.Elem2id[ft]
		//             this.Weight[tid][ftid] += alpha * (*_diag)[ftid] * val
		//             (*_diag)[ftid] -= (*_diag)[ftid] * val * beta * val * (*_diag)[ftid]
		//         }

		//         /* update parameters of predicted label */
		//         M = margins[argmax]
		//         _diag = &this.diag[argmax]
		//         Vpred := calcConfidence(_diag, &xi, &this.FtDict)
		//         _n = 1. + 2.*this.phi*M
		//         sqrt = math.Sqrt(math.Pow(_n, 2) - 8.*this.phi*(M-this.phi*Vpred))
		//         gamma = (-1.*_n + sqrt) / (4. * this.phi * Vpred)
		//         alpha = Max(0., gamma)
		//         beta = 2. * alpha * this.phi / (1. + 2.*alpha*this.phi*Vpred)
		//         for _, ft := range features {
		//             val := xi[ft]
		//             ftid := this.FtDict.Elem2id[ft]
		//             this.Weight[argmax][ftid] -= alpha * (*_diag)[ftid] * val
		//             (*_diag)[ftid] -= (*_diag)[ftid] * val * beta * val * (*_diag)[ftid]
		//         }
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
	diag  [][]float64
}

func NewArow(gamma float64) *Arow {
	arow := Arow{&Learner{}, 1., gamma, make([][]float64, 0, 100)}
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
		this.diag,
	}
	return &params
}

func (this *Arow) SetParams(params *[][][]float64) {
	this.Weight = (*params)[0]
	this.diag = (*params)[1]
}

func (this *Arow) Fit(x *[]map[string]float64, y *[]string) {
	for i := 0; i < len(*x); i++ {
		xi := (*x)[i]
		yi := (*y)[i]
		features := GetSortedFeatures(&xi)
		this.updateDict(&features, &yi)

		tid := this.LabelDict.Elem2id[yi]

		/* expand label size */
		if len(this.Weight) <= tid {
			for k := len(this.Weight); k <= tid; k++ {
				this.Weight = append(this.Weight, make([]float64, 0, 10000))
			}
		}
		if len(this.diag) <= tid {
			for k := len(this.diag); k <= tid; k++ {
				this.diag = append(this.diag, make([]float64, 0, 10000))
			}
		}

		argmax := -1
		max := math.Inf(-1)
		margins := make([]float64, len(this.Weight), len(this.Weight))
		for yid := 0; yid < len(this.Weight); yid++ {
			w := &this.Weight[yid]
			d := &this.diag[yid]
			dot := 0.
			for _, ft := range features {
				val := xi[ft]
				ftid := this.FtDict.Elem2id[ft]

				/* expand feature size */
				if len(*w) <= ftid {
					for k := len(*w); k <= ftid; k++ {
						*w = append(*w, 0.)
					}
				}
				if len(*d) <= ftid {
					for k := len(*d); k <= ftid; k++ {
						*d = append(*d, 1.*this.a)
					}
				}

				dot += (*w)[ftid] * val
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
				_diag := &this.diag[yid]
				V := calcConfidence(_diag, &xi, &this.FtDict)
				beta := 1. / (V + this.gamma)
				alpha := Max(0., 1.-sign*margins[yid]) * beta
				if alpha == 0. {
					continue
				}
				for ft, val := range xi {
					ftid := this.FtDict.Elem2id[ft]
					this.Weight[yid][ftid] += sign * alpha * (*_diag)[ftid] * val
					(*_diag)[ftid] -= beta * (*_diag)[ftid] * val * val * (*_diag)[ftid]
				}
			}
			/* The top-1 version of AROW algorithm */
			//             /* update parameters of true label */
			//             _diag := &this.diag[tid]
			//             Vtrue := calcConfidence(_diag, &xi, &this.FtDict)
			//             beta := 1. / (Vtrue + this.gamma)
			//             alpha := Max(0., 1.-margins[tid]) * beta
			//             if alpha > 0. {
			//                 for ft, val := range xi {
			//                     ftid := this.FtDict.Elem2id[ft]
			//                     this.Weight[tid][ftid] += alpha * (*_diag)[ftid] * val
			//                     (*_diag)[ftid] -= beta * (*_diag)[ftid] * val * val * (*_diag)[ftid]
			//                 }
			//             }

			//             /* update parameters of predicted label */
			//             _diag = &this.diag[argmax]
			//             Vpred := calcConfidence(_diag, &xi, &this.FtDict)
			//             beta = 1. / (Vpred + this.gamma)
			//             alpha = Max(0., 1.+margins[argmax]) * beta
			//             if alpha > 0. {
			//                 for ft, val := range xi {
			//                     ftid := this.FtDict.Elem2id[ft]
			//                     this.Weight[argmax][ftid] -= alpha * (*_diag)[ftid] * val
			//                     (*_diag)[ftid] -= beta * (*_diag)[ftid] * val * val * (*_diag)[ftid]
			//                 }
			//             }
			//         }
		}
	}
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

func calcConfidence(diag *[]float64, x *map[string]float64, ftdict *Dict) float64 {
	V := 0.
	for ft, val := range *x {
		ftid := (*ftdict).Elem2id[ft]
		V += val * val * (*diag)[ftid]
	}
	return V
}

func GetSortedFeatures(x *map[string]float64) []string {
	features := make([]string, len(*x), len(*x))
	k := 0
	for name, _ := range *x {
		features[k] = name
		k++
	}
	sort.Strings(features)
	return features
}
