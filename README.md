So far, transformers have shown poor performance in arithmetic tasks, apparently due to the inability to track the exact position of each digit in a large dimension. In all the works I have reviewed, Accuracy (exact match with the answer) was chosen as a metric, so in my work I will also report this metric.


First, I wanted to focus on paper [[1]](#1), where the authors experiment with different ways of representing numbers and the ability of models to extrapolate the result. For extrapolation experiments, the models are trained on up to 50-digit numbers and evaluated on 60- digit numbers. The training was done in 100K iterations using batches of 128 examples. The encoder-decoder T5 models were trained. The following results were obtained:

```
| Model | Interpolation | Extrapolation |
| ------------- | ------------- | ------------- |
| T5-60M | 0.998 | 0.004 |
| T5-220M | 1.000 | 0.862 |
| T5-770M | 0.999 | 0.442 |
| T5-3B | 1.000 | 0.988 |

```
Authors conclude that regardless of the number of parameters and training examples, models cannot seem to learn addition rules that are independent of the length of the numbers seen during training. Models cannot extrapolate, i.e., they fail to perform simple arithmetic when evaluated on inputs whose length distribution differs from the one seen during training. This appears to be a problem that neither larger models, more compute, nor more data can solve.




But then I came across a recently published [[2]](#2). The authors propose a new approach to presenting information about the position of digits in a number.

*We find that training on only 20 digit numbers with a single GPU for one day, we can reach state-of-the-art performance, achieving up to 99% accuracy on 100 digit addition problems.*

They propose to use embeddings for this, they call them "Abacus Embeddings". Each digit is tokenized separately and the embedding of the digit is added to the relative position embedding. The embedding index is equal to the position of the digit in the number. For example:
```
Least Significant Digit First: 1  2  3  4  +  1  2  3  4  =  2  4  6  8
 Most Significant Digit First: 4  3  2  1  +  4  3  2  1  =  8  6  4  2
            Abacus Embeddings: 1  2  3  4  0  1  2  3  4  0  1  2  3  4
          Absolute Embeddings: 1  2  3  4  5  6  7  8  9  10 11 12 13 14
```
For extrapolation, during training, a random number is added to the embedding index - U[1;k], where k is a hyperparameter. That is, all positions are shifted by this number.

Also, based on the works [[3](#3), [4](#4), [5](#5)] inputs are formatted least significant digit first, e.g. 98282 + 3859172 = 2787472. Authors do not add any padding between digits and do not pad any numbers with zeros, neither in the case of carry digits, nor to make all operands the same length. For training, 20 million examples are generated, i.e. 50,000 examples for each pair of lengths (i, j), where i is the length of the first operand, j is the length of the second operand, i, j <= 20. Accuracy is also chosen as the reported metric. Three samples serve as a test: in distribution where the models are tested on problems up to the maximum size seen during training; out of distribution where the models are tested on problems greater than the maximum size seen during training but both operands are at most 100 digits; and extreme out of distribution where the models are tested on problems where both operands are of the same length and are both more than 100 digits and less than 160 digits.

Three architectures are used as models: standard autoregressive transformer model where multiple decoder layers are stacked in a feedforward manner, standard autoregressive transformer with input injection, where the embedded inputs are added to the input of each decoder layer and looped transformer. The transformer has 122 million parameters. 

Training was conducted on Nvidia RTXA4000 GPU for 24 hours. The following results were obtained for the standard transformer: 1.0 for in distribution, 0.979 for out of distribution and 0.306 for extreme out of distribution.



I based my work on the second article, because this approach demonstrates the best result. My main limiting factor was computing power (all work was done on the kaggle platform - GPU P100 with a time limit). For training data I took a slice of 5 million, numbers with a maximum of 15 digits. For testing I used 10000 data: 25 for each pair of lengths, lengths up to 20 digits inclusive. This data is taken from the training data of the article. 

My first approach was with the opt-125m and opt-350m models, in which I added Abacus Embeddings to the embedding layer and removed Absolute Embeddings.Inputs are formatted least significant digit first, as in the article. I chose the standard transformer architecture because, according to the experiments in the article, this architecture loses only when the training sample operands consist of a maximum of 10 digits; in other cases, it demonstrates either comparable or superior quality. But the training takes a very long time to converge, and there were experiments with different k, optimizers. Due to the quota, I could not complete this experiment. I decided to try a larger model with RoPE. Unfortunately, the authors do not measure the result with these embeddings on a standard transformer. The choice fell on the state of the art decoder with 464 million parameters NuExtract-tiny. But here I was again faced with time constraints and an expiring quota. That's why I decided to train this model without Abacus Embeddings, of course, in this case there is no need to even think about extrapolation. The results for each operand length are presented at the end of the notebook. Good results are obtained at lengths shorter than 15 and none at lengths longer, which is expected, although I could not find an article where measurements are taken for this setup. However, RoPE does limit the length generalization as models are trained only using rotations based on training data length. For 3.5 hours of training with batch size 8, I am happy with the result.


## References

<a id="1">[1]</a> 
Rodrigo Nogueira, Zhiying Jiang & Jimmy Lin. Investigating the Limitations of Transformers with Simple Arithmetic Tasks, 2021. https://arxiv.org/abs/2102.13019

<a id="2">[2]</a> 
Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, et al. Transformers Can Do Arithmetic with the
Right Embeddings. arXiv preprint arXiv:2405.17399, 2024. https://arxiv.org/pdf/2405.17399

<a id="3">[3]</a> 
Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio,
and Preetum Nakkiran. What algorithms can transformers learn? a study in length generalization.
arXiv preprint arXiv:2310.16028, 2023. https://arxiv.org/pdf/2310.16028

<a id="4">[4]</a> 
Ruoqi Shen, Sébastien Bubeck, Ronen Eldan, Yin Tat Lee, Yuanzhi Li, and Yi Zhang. Positional
description matters for transformers arithmetic. arXiv preprint arXiv:2311.14737, 2023. https://arxiv.org/pdf/2311.14737

<a id="5">[5]</a> 
Nayoung Lee, Kartik Sreenivasan, Jason D Lee, Kangwook Lee, and Dimitris Papailiopoulos.
Teaching arithmetic to small transformers. arXiv preprint arXiv:2307.03381, 2023. https://arxiv.org/pdf/2307.03381




