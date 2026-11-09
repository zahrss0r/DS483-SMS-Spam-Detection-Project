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

## Part 1 (Student 2): TF-IDF vectors 

In this part, we changed the cleaned SMS messages into numbers using TF-IDF. We used TfidfVectorizer with min_df=2, so words that appear in fewer than 2 messages were removed. We fitted the vectorizer only on the train data, which has 4136 messages, and then used it to transform both train and test. This avoids data leakage. The train matrix has shape (4136, 2643) and the test matrix has shape (1035, 2643), so the vocabulary has 2643 words. The train set has 3614 ham and 522 spam messages, and the test set has 904 ham and 131 spam messages. We also checked the top 5 words of one spam message. Its highest weights were moneynum (0.308), suppli (0.294), choic (0.294), worth (0.260), and cd (0.257), which are typical prize and money words. The ham message “U GOIN OUT 2NITE?” is very short, so it has only two weighted words: numnit (0.738) and goin (0.675).


##  Part 2 (Student 3): cosine similarity search

To evaluate the semantic clustering of our spam detection dataset, we implemented a cosine similarity search pipeline over the TF-IDF feature space of the training partition. By analyzing the vector representations of representative spam and ham messages, the system successfully identified nearest neighbors that share identical phrasing, promotional structures, or conversational contexts. The resulting similarity scores confirm that vectorized text representations effectively group similar linguistic patterns together, providing a robust foundation for downstream neighborhood-based classification or anomaly detection.

## Part 3 (Student 4): similarity in and between classes

We checked how similar the messages are using cosine similarity on the TF-IDF vectors of the train set (522 spam and 3614 ham messages). The average similarity between two spam messages is 0.1034, which is about 10 times higher than between two ham messages (0.0107). The spam-ham average is 0.0126, so a spam message is not close to ham messages. This shows that spam messages are more similar to each other than ham messages. A reason is that spam messages often use the same style and words, while ham messages are personal and use many different words. The top terms support this. For spam, the highest average TF-IDF terms are num (0.2516), call (0.0732), moneynum (0.0670), free (0.0509), txt (0.0371), claim (0.0308) and prize (0.0295). For ham, the top terms are common daily words like im (0.0229), go (0.0203), ok (0.0189) and get (0.0181), and their values are much lower. Overall, spam has a clear and repeated vocabulary, so it should be easier for a classifier to separate.

### Part 4 (Student 1): Task 2 results and explanation

TF-IDF gives each word a weight. The weight is high when a word appears in a message but is rare in the other messages, so common words get a low weight. In spam, the words with the highest average TF-IDF were num (0.25), call (0.07), moneynum (0.07) and free (0.05). moneynum means a money amount like £100 and nump means a price like 150p, so numbers and prices are typical for spam. In ham, the top words were normal chat words like im, go and ok.

Cosine similarity compares the TF-IDF vectors of two messages. It is 1 when they use the same words and 0 when they share no words. Spam messages were much more similar to each other (0.10) than ham messages (0.01). The reason is that spam often comes in campaigns with almost the same text. For example, one spam query had a match with 0.93 similarity, and 4 of its top 5 matches were spam. For all 3 ham queries, the top 5 matches were all ham.

TF-IDF has a limit: it only counts words and does not understand word order or meaning. For example, one spam message about sending anonymous messages was closest to a ham message (0.53), only because both used the word "message".

## Task 3: Text Classification

### Week 3 Part 1 (Student 3): evaluation function and results table

This part establishes the shared evaluation framework for our classification tasks by importing timing and metrics utilities, initializing a global results list, and defining reusable functions to compute standard classification metrics (evaluate_model), format and display the summary table (show_results), and plot labeled confusion matrices (plot_cm). Finally, it trains and evaluates a baseline DummyClassifier that always predicts the majority class ("ham"), populating the initial row in our results table to serve as a benchmark for subsequent machine learning models.



---

## Task 4: Sequence-Based Classification

_To be added in Week 4._

---

## Task 5: Transformer-Based Classification

_To be added in Week 5._
