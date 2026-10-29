# DS483 Project Report Draft: SMS Spam Detection

This is the working draft of the report. Each student adds their paragraph in their week.
At the end, the final text is copied into the university template (Times New Roman, English only, tables as text).

---

## Task 1: Text Preprocessing

### Part 1: Dataset, duplicates, class distribution and data split

We used the SMS Spam Collection dataset from UCI. It has 5,574 SMS messages, and each message is labeled as ham (normal message) or spam. There are no missing values in the data.

We found 403 duplicate messages (309 ham and 94 spam). Some messages repeat a lot, for example "Sorry, I'll call later" appears 30 times. We removed the duplicates and kept only one copy of each message. We did this because if the same message is in the train data and the test data, the model already saw it, and the results will look better than they really are. After removing them we have 5,171 messages.

The data is not balanced. We have 4,518 ham messages (87.37%) and only 653 spam messages (12.63%). So if a model says "ham" for every message, it gets about 87% accuracy but it does not find any spam. This is why we will not use accuracy alone, we will also use precision, recall and F1-score.

We split the data into 80% train (4,136 messages) and 20% test (1,035 messages). We used stratify so both parts have almost the same percent of spam (12.62% in train and 12.66% in test). We also used random_state = 42 so we get the same split every time we run the notebook. All the models in the next tasks will use this same test data, so the comparison between them is fair.

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
