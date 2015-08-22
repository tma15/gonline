package gonline

import (
	"sync"
)

func FitLearners(learners *[]LearnerInterface, x *[]map[string]float64, y *[]string) {
	var wg sync.WaitGroup
	//     var mu sync.Mutex
	num_learner := len(*learners)
	num_data := len(*x)
	buffer := make(chan int, num_learner)
	sizechunk := num_data/num_learner + 1
	for i := 0; i < num_learner; i++ {
		wg.Add(1)
		go func(ch chan int) {
			//             mu.Lock()
			//             defer mu.Unlock()
			defer wg.Done()
			for j := range ch {
				start := j * sizechunk
				end := (j + 1) * sizechunk
				if end >= num_data {
					end = num_data - 1
				}
				x_j := (*x)[start:end]
				y_j := (*y)[start:end]
				(*learners)[j].Fit(&x_j, &y_j)
			}
		}(buffer)
	}
	for i := 0; i < num_learner; i++ {
		buffer <- i
	}
	close(buffer)
	wg.Wait()
}

func AverageModels(learners *[]LearnerInterface) {
	avg_w := make([][]float64, 10, 10)
	avg_ftdic := NewDict()
	avg_labeldic := NewDict()

	for _, learner := range *learners {
		w := learner.GetParam()
		ftdict, labeldict := learner.GetDics()
		for yid := 0; yid < len(*w); yid++ {
			y := labeldict.Id2elem[yid]
			if !avg_labeldic.HasElem(y) {
				avg_labeldic.AddElem(y)
			}
			yid_avg := avg_labeldic.Elem2id[y]
			for i := len(avg_w); i <= yid_avg; i++ {
				avg_w = append(avg_w, make([]float64, 0, 1000))
			}
			for ftid := 0; ftid < len((*w)[yid]); ftid++ {
				ft := ftdict.Id2elem[ftid]
				if !avg_ftdic.HasElem(ft) {
					avg_ftdic.AddElem(ft)
				}
				ftid_avg := avg_ftdic.Elem2id[ft]
				for i := len(avg_w[yid_avg]); i <= ftid_avg; i++ {
					avg_w[yid_avg] = append(avg_w[yid_avg], 0.)
				}
				avg_w[yid_avg][ftid_avg] += (*w)[yid][ftid] / float64(len(*learners))
			}
		}
	}

	var wg sync.WaitGroup
	num_learner := len(*learners)
	buffer := make(chan int, num_learner)
	for i := 0; i < num_learner; i++ {
		wg.Add(1)
		go func(ch chan int) {
			defer wg.Done()
			for j := range ch {
				(*learners)[j].SetParam(&avg_w)
				(*learners)[j].SetDics(&avg_ftdic, &avg_labeldic)
			}
		}(buffer)
	}
	for i := 0; i < num_learner; i++ {
		buffer <- i
	}
	close(buffer)
	wg.Wait()
}
