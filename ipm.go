package gonline

import (
	//     "fmt"
	//     "os"
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

func average_two(learner1, learner2 *LearnerInterface) *LearnerInterface {
	params := (*learner1).GetParams()
	num_params := len(*params)
	avg_params := make([][][]float64, num_params, num_params)
	for i := 0; i < num_params; i++ {
		avg_params = append(avg_params, make([][]float64, 10))
	}

	avg_ftdic := NewDict()
	avg_labeldic := NewDict()
	learners := []LearnerInterface{*learner1, *learner2}
	for _, learner := range learners {
		params := learner.GetParams()
		ftdict, labeldict := learner.GetDics()
		for yid := 0; yid < len(labeldict.Id2elem); yid++ {
			y := labeldict.Id2elem[yid]
			if !avg_labeldic.HasElem(y) {
				avg_labeldic.AddElem(y)
			}
			yid_avg := avg_labeldic.Elem2id[y]
			for p := 0; p < num_params; p++ {
				for i := len(avg_params[p]); i <= yid_avg; i++ {
					avg_params[p] = append(avg_params[p], make([]float64, 0, 1000))
				}
			}
			for p := 0; p < num_params; p++ {
				for ftid := 0; ftid < len((*params)[p][yid]); ftid++ {
					if ftid >= len((*params)[p][yid]) {
						continue
					}
					ft := ftdict.Id2elem[ftid]
					if !avg_ftdic.HasElem(ft) {
						avg_ftdic.AddElem(ft)
					}
					ftid_avg := avg_ftdic.Elem2id[ft]
					for i := len(avg_params[p][yid_avg]); i <= ftid_avg; i++ {
						avg_params[p][yid_avg] = append(avg_params[p][yid_avg], 0.)
					}
					avg_params[p][yid_avg][ftid_avg] += (*params)[p][yid][ftid] / float64(len(learners))
				}
			}
		}
	}
	(*learner1).SetParams(&avg_params)
	(*learner1).SetDics(&avg_ftdic, &avg_labeldic)
	return learner1
}

/*
	Repeat following processes:
	  1. For every two learners, calculate average model,
	  2. generate a slice of averaged models.
	Finally, return an average model over all learners.
*/
func AverageModels(learners []LearnerInterface) *LearnerInterface {
	if len(learners)%2 != 0 { /* add learner to make length of learners is even number */
		learners = append(learners, learners[len(learners)/2])
	}
	num_learner := len(learners)
	buffer := make(chan int, num_learner)
	results := make(chan *LearnerInterface, num_learner)

	var wg sync.WaitGroup
	for i := 0; i < num_learner; i++ {
		wg.Add(1)
		go func(ch chan int) {
			defer wg.Done()
			for j := range ch {
				//                 fmt.Println(j, j+num_learner/2)
				l1 := learners[j]
				l2 := learners[j+num_learner/2]
				l_avg := average_two(&l1, &l2)
				results <- l_avg
			}
		}(buffer)
	}
	for i := 0; i < num_learner/2; i++ {
		buffer <- i
	}
	close(buffer)
	wg.Wait()
	close(results)
	learners_avg := make([]LearnerInterface, 0, num_learner/2)
	for l_avg := range results {
		learners_avg = append(learners_avg, *l_avg)
	}

	if len(learners_avg) == 1 {
		return &learners_avg[0]
	}
	return AverageModels(learners_avg)
}

func BroadCastModel(avg_learner *LearnerInterface, learners *[]LearnerInterface) {
	avg_w := (*avg_learner).GetParam()
	avg_ftdic, avg_labeldic := (*avg_learner).GetDics()
	num_learner := len(*learners)
	var wg sync.WaitGroup
	buffer := make(chan int, num_learner)
	for i := 0; i < num_learner; i++ {
		wg.Add(1)
		go func(ch chan int) {
			defer wg.Done()
			for j := range ch {
				(*learners)[j].SetParam(avg_w)
				(*learners)[j].SetDics(avg_ftdic, avg_labeldic)
			}
		}(buffer)
	}
	for i := 0; i < num_learner; i++ {
		buffer <- i
	}
	close(buffer)
	wg.Wait()
}
