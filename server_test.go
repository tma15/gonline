package gonline

import (
	"fmt"
	"sync"

	"testing"
)

func BenchmarkDistributedFitting(b *testing.B) {
	hosts := []string{"127.0.0.1"}
	ports := []string{"8888"}
	//     hosts := []string{"127.0.0.1", "127.0.0.1"}
	//     ports := []string{"8888", "8889"}
	//     hosts := []string{"127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"}
	//     ports := []string{"8888", "8889", "8890", "8891"}

	numServers := len(hosts)
	for i := 0; i < numServers; i++ {
		s := NewLearnerServer(hosts[i], ports[i])
		go s.Start()
	}

	x, y := LoadData("/Users/makino/go/src/github.com/tma15/gonline/gonline/news20.scale.big")
	data := Data{
		X: x,
		Y: y,
	}

	num_data := len(*data.X)
	sizechunk := num_data/numServers + 1

	cli := NewClient()

	var avg_learner *LearnerInterface
	numLoop := 1
	for t := 0; t < numLoop; t++ {
		learners := make(chan *LearnerInterface)
		buffer := make(chan int)
		var wg sync.WaitGroup
		wg.Add(1)
		go func(procs chan int) {
			defer wg.Done()
			for pid := range procs {
				start := pid * sizechunk
				end := (pid + 1) * sizechunk
				if end >= num_data {
					end = num_data - 1
				}
				data_batch := data.GetBatch(start, end)

				learner := cli.SendData(hosts[pid], ports[pid], data_batch)
				learners <- learner
			}
		}(buffer)

		go func() {
			for i := 0; i < numServers; i++ {
				buffer <- i
			}
			close(buffer)
			wg.Wait()
			close(learners)
		}()

		_learners := make([]LearnerInterface, numServers, numServers)
		i := 0
		for learner := range learners {
			_learners[i] = *learner
			i++
		}
		avg_learner = AverageModels(_learners)

		buffer2 := make(chan int)
		var wg2 sync.WaitGroup
		wg2.Add(1)
		go func(procs chan int) {
			defer wg2.Done()
			for pid := range procs {
				cli.SendModel(hosts[pid], ports[pid], avg_learner)
			}
		}(buffer2)
		go func() {
			for i := 0; i < numServers; i++ {
				buffer2 <- i
			}
			close(buffer2)
			wg2.Wait()
		}()
	}

	(*avg_learner).Save("model")

	cls := LoadClassifier("model")

	numCorr := 0
	numTotal := 0
	x, y = LoadData("/Users/makino/go/src/github.com/tma15/gonline/gonline/news20.t.scale")
	for i, x_i := range *x {
		j := cls.Predict(&x_i)
		if cls.LabelDict.Id2elem[j] == (*y)[i] {
			numCorr += 1
		}
		numTotal += 1
	}
	acc := float64(numCorr) / float64(numTotal)
	fmt.Printf("test accuracy: %f (%d/%d)\n", acc, numCorr, numTotal)

}
