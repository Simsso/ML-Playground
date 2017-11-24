# Online Hard Example Mining
Inspired by Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick's [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540) (2016), this package explores the effects of backpropagation with the top _k_ batch entries only. The top _k_ entries are the ones that induce the highest loss and consequently affect the gradient the most. The following figures show, how the algorithm has performed given a batch size _n_ from which gradients were computed using backpropagation and solely the top _k_ elements.

For simplicity, the neural net is trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) data set. The network has one hidden layer with 1024 units. For more details have a look into the implementation (model.py). 

## Results
Depending of the ratio between _n_ and _k_, the accuracy grew faster or slower compared to _n=k_ (default mode of operation). Consistently, the backpropagation loss grows with _n:k_.

| Color        | Batch size _n_  | _k_        | Notes |
| ------------ |-----------------| -----------|-------|
| Light blue   | 512             | 512        | Default mode of operation. |
| Red          | 1024            | 512        | Greater batch size, same number of samples for backpropagation. |
| Cyan         | 512            | 384        | Balanced _n:k_ ration where _n_ is chosen to match the default mode of operation. |
| Purple       | 1024            | 128        | Great _n:k_ reation. |

All results are relative to the default mode of operation, i.e. _n_ is a given number and _k=n_. When choosing a greater batch size and performing backpropagation with the top 50% entries, better accuarcy results were achieved (red). This comparison is not entirely _fair_ though, as the batch size is greater during the forward pass. 

Cyan is using the same forward pass batch size and backpropagates with _3/4_ of the samples (_n:k_=1.33). This yields slightly better results than the default mode of operation and is the key finding of these experiments: A well balanced _n:k_ ratio can yield better results than the naive implementation _n=k_, without additional computational cost. However, the ratio is problem specific and needs to be fine-tuned. Shrivastava et al. go with higher values (_n:k=15.6_?).

Choosing only the most extrem samples (purple) does not lead to consistent improvements, possibly because a decent junk of samples is dropped, whose contribution to the gradient would not have been negligible (see Fig. 2).

![scalar](https://lh3.googleusercontent.com/RayQHwSsJwwUjg9c403WK5OHYahQXiKJWBZwnW53cft8Lt38vD37PcyxV2xyVVEgax-42HBSD7wRmm7PCAzo4OLx0YaD3R7on14YdjFJWs3sFH2fFyUsxoDYnr6Mmg90IvYFrfwhEM6CfjMam4zbqMxwIp5CXB45zBUjfG0JY6bxadv1sH2THvKI1DF9Fe6wP5qSQRBadp5GR5qyFT7OfvmdDyfsrcMkwqXnJ3E5rpamDuhC_zLsLwy77NZF0jkkeEa9ChrRmNFeQdK5A2RzVnkRq1qE8b0SwhqzqlKsmuxS23t5Wavt4yI1kEMSeDYLQiuSL0-AWclqq5h0N6aU1BrVlv8-1UCGa_fRpxVRct1lfQO3JLYqfNRWzEpndsoA6wYpVDz_jnzHRswyh4GegTub_NnKX6i_4pZkW7udBh4kzi7qD3VpsRFKDbiLTsbA9GYp4hIyHqKiqKd2gZJP8pngUiD86aWyxB21za6l1o88bbaXzWoPMukOM8CNls5bR2Kd4oq0TqOCricRCsSMWaWXgMXQdZtFXGY3WfFczUr-kdosu12UZ2plk68QIPcqu6nL9I6xplIF7IhIPdWeK5D3NPkqZB4nZGUO_ZurJ7fdq9IHvKPLypVBqX3LOm-Fl_Wpks0t3sDjt1yfbXIm7tMOz_-sgeL0GIWn=w1440-h690-no)

Fig. 1: The **accuracy** is the number of correct class predictions (conversion of softmax prediction to one-hot matches labels). The **L2 loss** is the mean loss of samples chosen for backpropagation. For great _n:k_ ratios, this value consistently remains at a high level. Depending on the network's size, this might not necessarily be the case.

![hist](https://lh3.googleusercontent.com/MkdJLErz5PBlTN7BiXAb_Agt3P5GXJmb2_MKBXf2IFp1k3evahMQQmTuhnyzwMDV-rpb8woAhy1eZ8OD4rvxnbNdL22uhd-KR5CTU4OCqUi8x-bBTQeXqavLVSf9JBYBdtUD2OnucFAh_QQNl0cp29sUUky1KNtl9uXSmpvbQm_KXMuVl616cup6gfgb_SXqNec94vDw9fA9nMHFAo7CbxmM0VlFaGkK2wuNEa4pUD1Wd_dLB5gjevTotR1ojVALGwjoaNOb2yUWXOXiGRiF6wBzLJatJbUgybYZGrK_DflUpNn8LOjBvyEkf1iOWbGM71KmdaY5JBEc1PZeC1jPjm8WuowAbF0cdo_HLWd-9eykch1pnTMsoKX9m1ZqnChKcKnFaHnQCDazjzlIoS4NXcrXOUVe2MivXzjWIE-edHIJGsXJFp8kz7800-2F_0Pjte3XvIZyfOIUAPwLNJCh_pvjIJRlqzg1XgAz-fw-ssmiCN73kNgC_zXA5o8JaRHVE2vORtaPxvQeQgeXC4DO5HG8tlIlT08r0jgjksmg_SScJEACjyh86YA4bbY-qZDCUgtVuDlw0-aQj_1yidkdt5tkGYHT9EjPLSB-KUNLJBNKu27L2BE96E72Yr7nGNQTlzD01WuTADXFAVIARmTzgtBX2GPxXr2BZzaN=w1316-h737-no)

Fig. 2: The histograms show the value distribution of the loss of samples in a feedforward pass batch (size _n_, name **loss across batch**) and a backpropagation batch (size _k_, name **loss across backpropagation batch**). For high _n:k_ ratios, the backpropagation batch loss value distribution is in an unhealthy area (top left histogram; _n:k=8_). Values lower than 0.28 are capped. Ratios that keep _k_ as low as possible but the loss histogram in a _typical_ shape (top right; _n:k=1.33_) are desirable.