# DS483 Project Report Draft: SMS Spam Detection

This is the working draft of the report. Each student adds their paragraph in their week.
At the end, the final text is copied into the university template (Times New Roman, English only, tables as text).

---

## Task 1: Text Preprocessing

### Part 1 (Student 1): Dataset, duplicates, class distribution and data split

We used the SMS Spam Collection dataset from the UCI Machine Learning Repository. It has 5,574 SMS messages. Each message has one label: ham (a normal message) or spam (an unwanted message). There were no missing values.

We found 403 duplicate messages (309 ham and 94 spam). Some messages appear many times. For example, "Sorry, I'll call later" appears 30 times. We removed the duplicates and kept the first copy of each message. We did this because a message that is in both the training set and the test set makes the results look better than they really are. After this step, 5,171 messages remained.

The data is not balanced. There are 4,518 ham messages (87.37%) and 653 spam messages (12.63%). This means a model that always says "ham" would get about 87% accuracy and still miss every spam message. So accuracy alone is not enough, and in Tasks 3 to 5 we also use precision, recall and F1-score for the spam class.

We split the data once into 80% training (4,136 messages) and 20% testing (1,035 messages). We used a stratified split, so both sets keep the same share of spam (12.62% in training and 12.66% in testing). We fixed the random seed (random_state = 42), so the split is the same every time. All models in this project use this same test set, so the comparison between them is fair.

**Table 1. Class distribution after removing duplicates**

| Class | Messages | Percent |
| --- | --- | --- |
| Ham | 4,518 | 87.37% |
| Spam | 653 | 12.63% |
| Total | 5,171 | 100% |

**Table 2. Train and test split**

| Set | Messages | Ham | Spam | Spam % |
| --- | --- | --- | --- | --- |
| Train | 4,136 | 3,614 | 522 | 12.62% |
| Test | 1,035 | 904 | 131 | 12.66% |

### Part 2 (Student 2): Text characteristics of ham and spam

_To be added in Week 1._

### Part 3 (Student 3): Regular expressions, tokenization, case and punctuation

_To be added in Week 1._

### Part 4 (Student 4): Stopwords, stemming and examples before and after

_To be added in Week 1._

---

## Task 2: Vector Representation and Similarity

_To be added in Week 2._

---

## Task 3: Text Classification

_To be added in Week 3._

---

## Task 4: Sequence-Based Classification

_To be added in Week 4._

---

## Task 5: Transformer-Based Classification

_To be added in Week 5._
