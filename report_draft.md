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

We studied the text features of ham and spam messages to find which ones can help tell them apart. We made a copy of the data called feat_df and added seven features: number of characters, number of words, number of digits, has a link (http or www), number of currency signs (£ $ €), percent of capital letters, and number of ! marks. To keep the test data unseen, we compared the two classes using only the train data (4,136 messages: 3,614 ham and 522 spam).

Spam messages are much longer than ham messages. The mean length is 137.74 characters for spam and 70.97 for ham, and spam has 23.69 words on average compared with 14.24 for ham. Most spam messages are close to 160 characters, which is the normal SMS limit, while ham messages are shorter and more spread out. The clearest difference is the number of digits: spam has 15.44 digits per message and ham has only 0.31, which is about 50 times less. This is because spam often has phone numbers, prices and prize codes.

Spam also has more currency signs, more exclamation marks and a higher percent of capital letters. Links are not common, but they appear almost only in spam: 14.9% of spam messages have a link, compared with 0.1% of ham. This means a link is a strong sign of spam, but most spam messages do not have one, so it cannot be used alone.

**Table 3. Mean of each feature in the train data**

| Feature | Ham mean | Spam mean |
| --- | --- | --- |
| Characters (char_len) | 70.967 | 137.741 |
| Words (word_len) | 14.241 | 23.690 |
| Digits (n_digits) | 0.309 | 15.444 |
| Has link (has_link) | 0.001 | 0.149 |
| Currency signs (n_currency) | 0.006 | 0.427 |
| Capital letters % (caps_ratio) | 5.663 | 10.766 |
| ! marks (n_exclaim) | 0.182 | 0.693 |

We also counted the 15 most common words in each class (lowercase, split on spaces). Ham messages use many personal words like “i”, “u”, “my” and “me”. Spam messages use words like “call”, “your”, “free” and “2”. Seven words, such as “to”, “a”, “you” and “the”, are in both lists, so they do not help to separate the classes. We will use stop word removal in the next steps to get clearer spam words.

**Table 4. Top 15 words in the train data (word count)**

| Rank | Ham word | Count | Spam word | Count |
| --- | --- | --- | --- | --- |
| 1 | i | 1,676 | to | 469 |
| 2 | you | 1,281 | a | 243 |
| 3 | to | 1,172 | call | 233 |
| 4 | the | 867 | your | 189 |
| 5 | a | 760 | you | 182 |
| 6 | and | 675 | for | 158 |
| 7 | u | 633 | the | 146 |
| 8 | in | 583 | or | 129 |
| 9 | my | 540 | free | 128 |
| 10 | is | 507 | 2 | 125 |
| 11 | me | 446 | is | 116 |
| 12 | for | 392 | ur | 105 |
| 13 | of | 385 | have | 91 |
| 14 | it | 357 | on | 91 |
| 15 | that | 347 | and | 87 |


### Part 3 (Student 3): Regular expressions, tokenization, case and punctuation

This part defines regex based cleaning and tokenization functions to normalize text preserving predictive spam indicators like URLs, money, and numbers while removing punctuation and casing applies them to the dataset, evaluates vocabulary reduction, and aligns with core NLP principles regarding feature sparsity, signal retention, and pipeline hygiene.


Sets up of specific contribution (Student 3, Week 1 Part 3).

Function Definitions: Imports Python's built-in regular expression library (re) and defines basic_clean(text) and tokenize(text).
The cleaning function follows the exact 7-step sequence (lowercase, apostrophe removal, URL replacement, currency replacement, number replacement, punctuation removal, and whitespace normalization) to normalize text while preserving spam-indicator signals like money and links.

Application & Comparison: Applies basic_clean across both train_df and test_df to create the required clean_basic column. It outputs a side-by-side comparison table of the first 5 messages so our group can visually inspect the transformation.

Vocabulary Analysis: Computes and prints the vocabulary size (number of unique tokens) before and after cleaning on the training split, illustrating how normalization reduces noise and shrinks the feature space.

#Updated only on the title of  ( Week 1 - Task 1: Text Preprocessing
Part 3 (Student 3): regex, tokens, lowercase, punctuation )

### Part 4 (Student 4): Stopwords, stemming and examples before and after

In this part we finished the text preprocessing. We downloaded the 198 NLTK English stopwords and made a Porter stemmer. Then we built the function full_clean(), which runs basic_clean, tokenize, stopword removal and stemming, and joins the words with spaces. We applied it to train_df and test_df and saved the result in the column clean_text. Nothing was fitted on the test set. The vocabulary in train_df became smaller at each step: 13,600 unique words in the raw message, 7,106 in clean_basic, and 5,794 in clean_text. This is about 57% fewer unique words overall. Only 4 of 4,136 train messages and 2 of 1,035 test messages became empty, for example “Where @” and “Can a not?”, because all their words were stopwords. The examples show that spam words like “win”, “free” and “claim” stay, while common words like “you” and “have” are removed. The LSTM and BERT models will not use clean_text. They will use clean_basic and the raw message.


## Task 2: Vector Representation and Similarity

### Part 1 (Student 2): TF-IDF vectors 

In this part, we changed the cleaned SMS messages into numbers using TF-IDF. We used TfidfVectorizer with min_df=2, so words that appear in fewer than 2 messages were removed. We fitted the vectorizer only on the train data, which has 4136 messages, and then used it to transform both train and test. This avoids data leakage. The train matrix has shape (4136, 2643) and the test matrix has shape (1035, 2643), so the vocabulary has 2643 words. The train set has 3614 ham and 522 spam messages, and the test set has 904 ham and 131 spam messages. We also checked the top 5 words of one spam message. Its highest weights were moneynum (0.308), suppli (0.294), choic (0.294), worth (0.260), and cd (0.257), which are typical prize and money words. The ham message “U GOIN OUT 2NITE?” is very short, so it has only two weighted words: numnit (0.738) and goin (0.675).


### Part 2 (Student 3): cosine similarity search

To evaluate the semantic clustering of our spam detection dataset, we implemented a cosine similarity search pipeline over the TF-IDF feature space of the training partition. By analyzing the vector representations of representative spam and ham messages, the system successfully identified nearest neighbors that share identical phrasing, promotional structures, or conversational contexts. The resulting similarity scores confirm that vectorized text representations effectively group similar linguistic patterns together, providing a robust foundation for downstream neighborhood-based classification or anomaly detection.

### Part 3 (Student 4): similarity in and between classes

We checked how similar the messages are using cosine similarity on the TF-IDF vectors of the train set (522 spam and 3614 ham messages). The average similarity between two spam messages is 0.1034, which is about 10 times higher than between two ham messages (0.0107). The spam-ham average is 0.0126, so a spam message is not close to ham messages. This shows that spam messages are more similar to each other than ham messages. A reason is that spam messages often use the same style and words, while ham messages are personal and use many different words. The top terms support this. For spam, the highest average TF-IDF terms are num (0.2516), call (0.0732), moneynum (0.0670), free (0.0509), txt (0.0371), claim (0.0308) and prize (0.0295). For ham, the top terms are common daily words like im (0.0229), go (0.0203), ok (0.0189) and get (0.0181), and their values are much lower. Overall, spam has a clear and repeated vocabulary, so it should be easier for a classifier to separate.

### Part 4 (Student 1): Task 2 results and explanation

TF-IDF gives each word a weight. The weight is high when a word appears in a message but is rare in the other messages, so common words get a low weight. In spam, the words with the highest average TF-IDF were num (0.25), call (0.07), moneynum (0.07) and free (0.05). moneynum means a money amount like £100 and nump means a price like 150p, so numbers and prices are typical for spam. In ham, the top words were normal chat words like im, go and ok.

Cosine similarity compares the TF-IDF vectors of two messages. It is 1 when they use the same words and 0 when they share no words. Spam messages were much more similar to each other (0.10) than ham messages (0.01). The reason is that spam often comes in campaigns with almost the same text. For example, one spam query had a match with 0.93 similarity, and 4 of its top 5 matches were spam. For all 3 ham queries, the top 5 matches were all ham.

TF-IDF has a limit: it only counts words and does not understand word order or meaning. For example, one spam message about sending anonymous messages was closest to a ham message (0.53), only because both used the word "message".

## Task 3: Text Classification

### Week 3 Part 1 (Student 3): evaluation function and results table

This part establishes the shared evaluation framework for our classification tasks by importing timing and metrics utilities, initializing a global results list, and defining reusable functions to compute standard classification metrics (evaluate_model), format and display the summary table (show_results), and plot labeled confusion matrices (plot_cm). Finally, it trains and evaluates a baseline DummyClassifier that always predicts the majority class ("ham"), populating the initial row in our results table to serve as a benchmark for subsequent machine learning models.

### Week 3 Part 2 (Student 4): train Naive Bayes and Logistic Regression

In this part we trained two classical models on the TF-IDF features: Multinomial Naive Bayes and Logistic Regression. Naive Bayes trained in 0.0125 seconds, while Logistic Regression took 0.2146 seconds, so Naive Bayes was about 17 times faster to train. Both models then made predictions for all 1,035 test messages very quickly: 0.0066 seconds for Naive Bayes and 0.0011 seconds for Logistic Regression. Naive Bayes predicted 111 messages as spam, and Logistic Regression predicted 118. Naive Bayes has 5,286 parameters (two values for each of the 2,643 words), and Logistic Regression has 2,644 (one coefficient per word plus one intercept). The Logistic Regression coefficients are easy to read. The strongest spam words include “num”, “moneynum”, “url”, “free”, “txt”, “claim” and “prize”. The strongest ham words include “ok”, “ill”, “home”, “work” and “later”, which are normal chat words. The next step is to evaluate both models and compare their scores.

### Part 3 (Student 1): Comparing the two models

We tested Naive Bayes and Logistic Regression on the same test data (1,035 messages: 904 ham and 131 spam). We also compared them with a baseline that always says "ham". The baseline has a high accuracy (87.34%), but its recall and F1 for spam are 0, because it never finds any spam. This shows why accuracy alone is not enough.

Both models are much better than the baseline. Logistic Regression got an accuracy of 98.16%, a precision of 0.97, a recall of 0.88 and an F1-score of 0.92. Naive Bayes got an accuracy of 97.87%, a precision of 0.99, a recall of 0.84 and an F1-score of 0.91.

The confusion matrices show the difference. Naive Bayes put only 1 ham message in the spam folder, but it missed 21 spam messages. Logistic Regression put 3 ham messages in the spam folder and missed 16 spam messages. A ham message in the spam folder is a bigger problem for the user, because they can miss something important. Still, Logistic Regression is a little better overall, because it finds more spam and makes very few mistakes on ham. Both models are very fast and train in less than a second.

**Table. Results on the test data**

| Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| Baseline (always ham) | 0.8734 | 0.0000 | 0.0000 | 0.0000 |
| Naive Bayes | 0.9787 | 0.9910 | 0.8397 | 0.9091 |
| Logistic Regression | 0.9816 | 0.9746 | 0.8779 | 0.9237 |
---

### Part 4 (Student 2) : wrong predictions and Task 3 write-up

We compared a baseline, Naive Bayes and Logistic Regression on 1035 test messages. The baseline that always predicts ham reached 0.8734 accuracy but found no spam, so accuracy alone is misleading. Naive Bayes reached 0.9787 accuracy and an F1 of 0.9091. Logistic Regression was slightly better with 0.9816 accuracy and an F1 of 0.9237, mainly because its recall was higher (0.8779 against 0.8397). Naive Bayes had the higher precision (0.9910) and trained faster. Together the models made mistakes on 29 messages, and 12 of these were wrong in both. Naive Bayes made 1 false positive and 21 false negatives, and Logistic Regression made 3 false positives and 16 false negatives. The strongest spam words were num, moneynum, url, txt and free. Spam with tricks like F.R.E.E was still caught, but the Logistic Regression spam probability dropped from 0.895 to 0.660. The few real messages that were blocked were short, so short informal messages may be treated unfairly. We choose Logistic Regression as the better model, but a spam filter should keep false positives low.

## Task 4: Sequence-Based Classification

### Part 1 (Student 4) : Sequence-Based Classification

In this part we prepared the data for the LSTM model. First, we split the 4,136 training messages into 3,722 for training (train_part) and 414 for validation (val_part). The split used stratify, so both sets keep the same spam share of 12.6%. The test set (1,035 messages) was not changed. We used the clean_basic column, because the LSTM reads words in order and needs stopwords and full words. The messages are short: the mean length is 15.6 words, and the 95% length is 33 words. Because of this, MAX_LEN = 40 keeps almost every message complete, although the longest message has 162 words. The TextVectorization layer was adapted on train_part only and learned a vocabulary of 6,742 words, which is lower than VOCAB_SIZE = 10,000. Each message became a sequence of 40 numbers, with zeros as padding at the end. The final shapes are (3722, 40), (414, 40) and (1035, 40).

### Part 2 (Student 1): Building and training the LSTM

We built an LSTM model with four layers. The Embedding layer turns each word into a vector of 64 numbers. The LSTM layer (64 units) reads the words in order, so it can learn patterns like "call now" or "you have won". The Dropout layer turns off 30% of the units during training to reduce overfitting. The last Dense layer gives one number between 0 and 1, which is the probability that the message is spam. The model has 673,089 parameters, and most of them (640,000) are in the Embedding layer.

We trained the model for 5 epochs with a batch size of 32, using the adam optimizer and binary cross-entropy loss. We used train_part (3,722 messages) for training and val_part (414 messages) for validation. Training took 27 seconds.

The model learned very fast. After the first epoch, the training loss went down from 0.19 to 0.05. The validation accuracy was high from the start (0.99) and ended at 0.9952 after epoch 5, with a validation loss of 0.038. The validation accuracy went down a little in epoch 4 (0.971), but it went back up in epoch 5. This small change is normal because the validation set is small. We do not see clear overfitting, because the final training loss (0.044) and validation loss (0.038) are close.

**Table. Training results for each epoch**

| Epoch | Train accuracy | Train loss | Validation accuracy | Validation loss |
| --- | --- | --- | --- | --- |
| 1 | 0.9374 | 0.1896 | 0.9903 | 0.0406 |
| 2 | 0.9879 | 0.0538 | 0.9903 | 0.0288 |
| 3 | 0.9944 | 0.0250 | 0.9903 | 0.0356 |
| 4 | 0.9957 | 0.0186 | 0.9710 | 0.0748 |
| 5 | 0.9866 | 0.0443 | 0.9952 | 0.0381 |

### Part 3 (Student 2): test the LSTM

We tested the LSTM on test_df (1,035 messages), which was not used for training or validation. The model gives one probability for each message, and we used a threshold of 0.5 to decide spam (1) or ham (0). The prediction took 2.68 seconds. The LSTM predicted 117 messages as spam.

The LSTM got an accuracy of 0.9845, a precision of 0.9915, a recall of 0.8855 and an F1 of 0.9355. The confusion matrix shows that 903 ham messages and 116 spam messages were correct. Only 1 ham message was wrongly marked as spam, but 15 spam messages were missed. This means the model is very careful (high precision), but it still misses about 11% of spam. In total, the LSTM made 16 errors. Most of the errors we looked at were spam messages that were very short or unusual, like "ringtoneking 84484", so the model has little context to use.

The LSTM has the best results in our table. Its F1 (0.9355) is higher than Logistic Regression (0.9237) and Naive Bayes (0.9091), and it made 16 errors, against 19 and 22. But the difference is small, only 3 fewer errors than Logistic Regression, so it may be partly luck. The LSTM is also much more expensive. Training took 18.84 seconds against 0.11 seconds for Logistic Regression, and it has 673,089 parameters against 2,644. For this small dataset, Logistic Regression is almost as good and much cheaper.

**Table. Test results of all models**

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Baseline (always ham) | 0.8734 | 0.0000 | 0.0000 | 0.0000 |
| Naive Bayes | 0.9787 | 0.9910 | 0.8397 | 0.9091 |
| Logistic Regression | 0.9816 | 0.9746 | 0.8779 | 0.9237 |
| LSTM | 0.9845 | 0.9915 | 0.8855 | 0.9355 |

**Table. Cost and errors of each model**

| Model | Errors | Train time (s) | Predict time (s) | Parameters |
| --- | --- | --- | --- | --- |
| Naive Bayes | 22 | 0.0115 | 0.0008 | 5,286 |
| Logistic Regression | 19 | 0.1140 | 0.0012 | 2,644 |
| LSTM | 16 | 18.8414 | 2.6757 | 673,089 |

**Table. Confusion matrix of the LSTM on the test set**

| | Predicted ham | Predicted spam |
| --- | --- | --- |
| True ham | 903 | 1 |
| True spam | 15 | 116 |


### Part 4 (Student 3): compare LSTM and Task 4 write-up

This part closes Task 4 by comparing our sequence-based LSTM model against classical bag-of-words baselines through five steps: adding a section header, calling show_results() to print the evaluation table, plotting a grouped bar chart of Precision, Recall, and F1-Score, inspecting test messages where Logistic Regression and the LSTM disagreed, and writing a final summary weighing predictive performance against parameter count and training time.

## Task 5: Transformer-Based Classification

### Part 1 (Student 1): Preparing DistilBERT

For Task 5 we used DistilBERT (distilbert-base-uncased). It is a pretrained model, which means it already learned English from a very large amount of text, so we only need to fine-tune it on our SMS data. DistilBERT is a smaller and faster version of BERT (about 66 million parameters instead of 110 million), and it keeps most of BERT's accuracy. We used a T4 GPU in Colab for this task.

We gave the model the raw message, not clean_basic or clean_text, because BERT has its own tokenizer and it learned from normal text. The tokenizer has a vocabulary of 30,522 tokens. It splits rare words into smaller pieces that it knows. For example, the slang word "lor" became "lo" and "##r". It also adds [CLS] at the start and [SEP] at the end of every message.

In train_part, the messages have 25 tokens on average, and 95% of them have 54 tokens or less. The longest message has 216 tokens. We chose BERT_MAX_LEN = 64, so only 60 of the 3,722 training messages (1.6%) were cut. Shorter messages were padded to 64 tokens, and the attention mask tells the model to ignore the padding. We used the same split as the LSTM: 3,722 messages for training, 414 for validation and 1,035 for testing.

### Part 2 (Student 2): fine-tune DistilBERT

In this part we fine-tuned DistilBERT (distilbert-base-uncased) for SMS spam detection. We used the raw messages and added a new classification layer for two classes, ham and spam. We trained for 2 epochs with batch size 16 and learning rate 2e-5. The model has 66,955,010 parameters, which is much more than our earlier models. We trained on 3,722 messages and used 414 messages for validation, so the test set stayed unseen. Training took only 49.5 seconds. The average training loss was 0.0658. The validation loss was 0.0296 after epoch 1 and 0.0233 after epoch 2. The validation loss went down in both epochs, so the model kept learning and we see no sign of overfitting. The average training loss is higher than the validation loss because it is an average over the whole training, including the first steps when the model was still learning. These losses are low, but we still need the final test on test_df to compare DistilBERT with the other models.


### Part 3 (Student 3): test BERT and final comparison

In the final evaluation on the unseen test set of 1,035 SMS messages, our fine-tuned DistilBERT model achieved the strongest overall classification performance across all five models. DistilBERT reached an accuracy of 0.9923, precision of 0.9920, recall of 0.9466, and an F1-score of 0.9688, making only 8 total classification errors—7 false negatives and just 1 false positive. Comparing all approaches, performance steadily improved from the majority-class baseline (F1: 0.0000) to Naive Bayes (0.9091), Logistic Regression (0.9237), LSTM (0.9453), and finally DistilBERT (0.9688). However, this superior accuracy comes with substantial computational trade-offs. DistilBERT contains 66,955,010 parameters and required 53.0744 seconds to train and 2.8842 seconds to predict, whereas Logistic Regression achieved a strong 0.9237 F1-score using only 2,644 parameters in 0.0863 seconds of training. Overall, while DistilBERT offers the highest spam detection precision and recall, lightweight linear models remain attractive when speed and low memory usage are critical.


### Part 4 (Student 4): final recommendation and full run

In this final part we compared all models on the same 1035 test messages. DistilBERT gave the best F1 (0.9688), followed by Logistic Regression (0.9237), Naive Bayes (0.9091) and LSTM (0.7262). The baseline that always says ham has an accuracy of 0.8734 but an F1 of 0, so accuracy alone is not enough. DistilBERT made 8 errors (1 false positive, 7 false negatives), while Logistic Regression made 19 (3 false positives, 16 false negatives). Both models were wrong on 7 messages; the examples we checked were all spam that looks like normal business messages, such as ringtone orders and ticket offers. The price of DistilBERT is high: it has about 67 million parameters against 2,644 for Logistic Regression, needs 54.7 seconds to train against 0.08 seconds, and takes 3.73 seconds per 1000 predictions against 0.0045. It also needs a GPU. We recommend DistilBERT when a GPU is available because it misses fewer spam messages, and Logistic Regression as a cheap backup.
