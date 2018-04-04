# Assignment 2

In the assignment, we use CNN to classify patches. The number of patches are the following:  
* train
    * adult_females -- 22239
    * adult_males -- 3334
    * subadult_male -- 5315
    * pups -- 9687
    * backgrounds -- 120000

* valid
    * adult_females -- 5560
    * adult_males -- 834
    * subadult_male -- 665
    * pups -- 2422
    * backgrounds -- 30000

* test
    * adult_females -- 7881
    * adult_males -- 1079
    * subadult_male -- 678
    * pups -- 3971
    * backgrounds -- 80000

In the training phase, we use data augumentation to increase the data. Since the dataset is highly unbalanced, we use class_weights to re-map the loss. The class weights are defined as:  

{'backgrounds': 2, 'subadult_males': 5, 'pups': 4, 'adult_females': 0, 'adult_males': 1, 'juveniles': 3}

{0: 1.2748347349012907, 1: 8.507953181272509, 2: 0.2362375, 3: 2.3276541587979307, 4: 2.9267499483791037, 5: 10.66936394429808}

When we calculate the loss, class weights are multiplied to the original weights.

Conclusions

This network works well for classifying the backgrounds. The reasons are that we have a lot of background patches and they are easier to classify. 
