# **Bias in machine learning**
> * Bias is a systematic error from an erroneous assumption in the machine learning algorithm’s modeling. The algorithm tends to systematically learn the wrong signals by not considering all the information contained within the data. 

![](https://miro.medium.com/max/1400/0%2AaAJWomAPaurHTlTV.jpeg)

* Bias in Machine Learning models could often lead to unexpected outcomes.



## **`Sampling bias:`**
>It is a bias in which a sample is collected in such a way that it does not represent the true distribution because of non-random reasons.
## Reducing sampling bias
>Sampling bias can be reduced by making the sample as close to the true distribution (production traffic in case of ml models) as possible.

*`Simple Random Sampling with replacement :`*

* One strategy is to use random sampling with replacement from production traffic.
* *Sampling without replacement cannot be used since the population will cease to reflect the true distribution after some draws.*

**Stratified random sampling :**

>It is a method of sampling that involves the division of a population into smaller sub-groups known as `strata`.

* Then simple random sampling with replacement is applied within each stratum.

* Members in each of these sub-groups should be distinct so that every member of all groups get equal opportunity to be selected using simple probability.

* This sampling method is also called proportional random sampling.

## **` bias:`**
>Data snooping in ML model training refers to the phenomenon where test data has been observed / used in some way while training thereby leading to low test set error but high error in production.

## Reducing Data snooping bias

* The best way to reduce chances of data snooping is to separate out train/ test / tune datasets before any pre-processing. 
* The test dataset should not be processed along with train & tune.

* Build tooling to record overlaps in train / test datasets. 
* Any overlap which arises due to non-random reasons should be investigated and removed.

* Build monitoring to detect overlaps between training and production data.
