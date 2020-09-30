


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
sns.set_style("whitegrid") # Plotting style
get_ipython().run_line_magic('matplotlib', 'inline # Plots show up in notebook')
np.random.seed(7) # seeding random number generator

csv = "1429_1.csv"
df = pd.read_csv(csv)
df.head(2)


# - We can potentially refine sentiment analysis with the reviews.text column, with the actual rating of reviews.doRecommend column (boolean)
# - We can also label each review based on each sentiment
#     - title can contain positive/negative information about review

# In[3]:


data = df.copy()
data.describe()


:


data.info()


# - Drop reviews.userCity, reviews.userProvince, reviews.id, and reviews.didPurchase since these values are floats

# - We need to clean up the name column by referencing asins (unique products) since we have 7000 missing values

# In[5]:


data["asins"].unique()


# In[7]:


asins_unique = len(data["asins"].unique())
print("Number of Unique ASINs: " + str(asins_unique))


# **Visualizing the distributions of numerical variables:**

# In[8]:


#data.hist(bins=50, figsize=(20,15)) # builds histogram and set the number of bins and fig size (width, height)
#plt.show()




# # 3 Split into Train/Test




# In[9]:


from sklearn.model_selection import StratifiedShuffleSplit#To use sklearn's `Stratified ShuffleSplit` class, we're going to remove all samples that have NAN in review score, then covert all review scores to `integer` datatype
print("Before {}".format(len(data)))
dataAfter = data.dropna(subset=["reviews.rating"]) # removes all NAN in reviews.rating
print("After {}".format(len(dataAfter)))
dataAfter["reviews.rating"] = dataAfter["reviews.rating"].astype(int)


# In[10]:


split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in split.split(dataAfter, dataAfter["reviews.rating"]): 
    strat_train = dataAfter.reindex(train_index)
    strat_test = dataAfter.reindex(test_index)


# **Check to see if train/test sets were stratified proportionately in comparison to raw data.**

# In[11]:


len(strat_train)


# In[12]:


strat_train["reviews.rating"].value_counts()/len(strat_train) # value_count() counts all the values based on column


# In[13]:


len(strat_test)


# In[14]:


strat_test["reviews.rating"].value_counts()/len(strat_test)


# # 4 Data Exploration (Training Set)

# In[15]:


reviews = strat_train.copy()
reviews.head(2)


# Next, we will explore the following columns:
# - asins
# - name
# - reviews.rating
# - reviews.doRecommend
# - (reviews.numHelpful - not possible since numHelpful is only between 0-13 as per previous analysis in Raw Data)
# - (reviews.text - not possible since text is in long words)
# 
# Also, we will explore columns to asins

# ## 4.1 names / ASINs

# In[16]:


len(reviews["name"].unique()), len(reviews["asins"].unique())


# In[17]:


reviews.info()


# Working hypothesis: there are only 35 products based on the training data ASINs
# - One for each ASIN, but more product names (47)
# - ASINs are what's important here since we're concerned with products. There's a one to many relationship between ASINs and names
# - A single ASIN can have many names due to different vendor listings
# - There could also a lot of missing names/more unique names with slight variations in title (ie. 8gb vs 8 gb, NAN for product names)

# In[18]:


reviews.groupby("asins")["name"].unique()


# <u>Note</u>: there are actually 34 ASINs with one of the product having 2 ASINs

# In[19]:


# Lets see all the different names for this product that have 2 ASINs
different_names = reviews[reviews["asins"] == "B00L9EPT8O,B01E6AO69U"]["name"].unique()
for name in different_names:
    print(name)


# In[20]:


reviews[reviews["asins"] == "B00L9EPT8O,B01E6AO69U"]["name"].value_counts()


# **Confirmed our hypothesis that each ASIN can have multiple names. Therefore we should only really concern ourselves with which ASINs do well, not the product names.**

# In[21]:


fig = plt.figure(figsize=(16,10))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
reviews["asins"].value_counts().plot(kind="bar", ax=ax1, title="ASIN Frequency")
np.log10(reviews["asins"].value_counts()).plot(kind="bar", ax=ax2, title="ASIN Frequency (Log10 Adjusted)") 
plt.show()


# - Based on the bar graph for ASINs, we see that certain products have significantly more reviews than other products, which may indicate a higher sale in those specific products
# - We also see that the ASINs have a "right tailed" distribution which can also suggest that certain products have higher sales which can correlate to the higher ASINs frequencies in the reviews
# - We also took the log of the ASINs to normalize the data, in order display an in-depth picture of each ASINs, and we see that the distribution still follows a "right tailed" distribution
# 
# **This answers the first question that certain ASINs (products) have better sales, while other ASINs have lower sale, and in turn dictates which products should be kept or dropped.**

# In[22]:


# Entire training dataset average rating
reviews["reviews.rating"].mean()


# ## 4.2 reviews.rating / ASINs

# In[23]:


asins_count_ix = reviews["asins"].value_counts().index
plt.subplots(2,1,figsize=(16,12))
plt.subplot(2,1,1)
reviews["asins"].value_counts().plot(kind="bar", title="ASIN Frequency")
plt.subplot(2,1,2)
sns.pointplot(x="asins", y="reviews.rating", order=asins_count_ix, data=reviews)
plt.xticks(rotation=90)
plt.show()


# - 1a) The most frequently reviewed products have their average review ratings in the 4.5 - 4.8 range, with little variance
# - 1b) Although there is a slight inverse relationship between the ASINs frequency level and average review ratings for the first 4 ASINs, this relationship is not significant since the average review for the first 4 ASINs are rated between 4.5 - 4.8, which is considered good overall reviews
# - 2a) For ASINs with lower frequencies as shown on the bar graph (top), we see that their corresponding average review ratings on the point-plot graph (bottom) has significantly higher variance as shown by the length of the vertical lines. As a result, we suggest that, the average review ratings for ASINs with lower frequencies are not significant for our analysis due to high variance
# - 2b) On the other hand, due to their lower frequencies for ASINs with lower frequencies, we suggest that this is a result of lower quality products
# - 2c) Furthermore, the last 4 ASINs have no variance due to their significantly lower frequencies, and although the review ratings are a perfect 5.0, but we should not consider the significance of these review ratings due to lower frequency as explained in 2a)
# 
# **<u>Note</u> that point-plot graph automatically takes the average of the review.rating data.**

# ## 4.3 reviews.doRecommend / ASINs

# In[24]:


plt.subplots (2,1,figsize=(16,12))
plt.subplot(2,1,1)
reviews["asins"].value_counts().plot(kind="bar", title="ASIN Frequency")
plt.subplot(2,1,2)
sns.pointplot(x="asins", y="reviews.doRecommend", order=asins_count_ix, data=reviews)
plt.xticks(rotation=90)
plt.show()


# - From this analysis, we can see that the first 19 ASINs show that consumers recommend the product, which is consistent with the "reviews.rating / ASINs" analysis above, where the first 19 ASINs have good ratings between 4.0 to 5.0
# - The remaining ASINs have fluctuating results due to lower sample size, which should not be considered
# 
# **<u>Note</u>: reviews.text will be analyzed in Sentiment Analysis.**

# # 5 Correlations

# In[25]:


corr_matrix = reviews.corr()
corr_matrix
# Here we can analyze reviews.ratings with asins


# In[26]:


reviews.info()


# In[27]:


counts = reviews["asins"].value_counts().to_frame()
counts.head()


# In[28]:


avg_rating = reviews.groupby("asins")["reviews.rating"].mean().to_frame()
avg_rating.head()


# In[29]:


table = counts.join(avg_rating)
table.head(30)


# In[30]:


plt.scatter("asins", "reviews.rating", data=table)
table.corr()



# # 6 Sentiment Analysis

# Using the features in place, we will build a classifier that can determine a review's sentiment.

# ## 6.1 Set Target Variable (Sentiments)

# Segregate ratings from 1-5 into positive, neutral, and negative.

# In[33]:


def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"
# Add sentiments to the data
strat_train["Sentiment"] = strat_train["reviews.rating"].apply(sentiments)
strat_test["Sentiment"] = strat_test["reviews.rating"].apply(sentiments)
strat_train["Sentiment"][:20]


# In[35]:


# Prepare data
X_train = strat_train["reviews.text"]
X_train_targetSentiment = strat_train["Sentiment"]
X_test = strat_test["reviews.text"]t
X_test_targetSentiment = strat_test["Sentiment"]
print(len(X_train), len(X_test))


# 27,701 training samples and 6926 testing samples.

# ## 6.2 Extract Features



# In[36]:


# Replace "nan" with space
X_train = X_train.fillna(' ')
X_test = X_test.fillna(' ')
X_train_targetSentiment = X_train_targetSentiment.fillna(' ')
X_test_targetSentiment = X_test_targetSentiment.fillna(' ')

# Text preprocessing and occurance counting
from sklearn.feature_extraction.text import CountVectorizer 
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train) 
X_train_counts.shape


# Here we have 27,701 training samples and 12,526 distinct words in our training sample.
# 
# 
# Also, with longer documents, we typically see higher average count values on words that carry very little meaning, this will overshadow shorter documents that have lower average counts with same frequencies, as a result, we will use **TfidfTransformer** to reduce this redundancy:
# - Term Frequencies (**Tf**) divides number of occurrences for each word by total number of words
# - Term Frequencies times Inverse Document Frequency (**Tfidf**) downscales the weights of each word (assigns less value to unimportant stop words ie. "the", "are", etc)

# In[37]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# ## 6.3 Building a Pipeline from the Extracted Features

# We will use Multinominal Naive Bayes as our Classifier
# - Multinominal Niave Bayes is most suitable for word counts where data are typically represented as **word vector counts** (number of times outcome number X[i,j] is observed over the n trials), while also ignoring non-occurrences of a feature i


# In[39]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
clf_multiNB_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_nominalNB", MultinomialNB())])
clf_multiNB_pipe.fit(X_train, X_train_targetSentiment)


# ## 6.4 Test Model

# In[40]:


import numpy as np
predictedMultiNB = clf_multiNB_pipe.predict(X_test)
np.mean(predictedMultiNB == X_test_targetSentiment)



# - Test other models
# - Fine tune the best models to avoid over-fitting

# ## 6.5 Testing Other Models

# **Logistic Regression Classifier**

# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
clf_logReg_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_logReg", LogisticRegression())])
clf_logReg_pipe.fit(X_train, X_train_targetSentiment)

import numpy as np
predictedLogReg = clf_logReg_pipe.predict(X_test)
np.mean(predictedLogReg == X_test_targetSentiment)


# **Support Vector Machine Classifier**

# In[42]:


from sklearn.svm import LinearSVC
clf_linearSVC_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_linearSVC", LinearSVC())])
clf_linearSVC_pipe.fit(X_train, X_train_targetSentiment)

predictedLinearSVC = clf_linearSVC_pipe.predict(X_test)
np.mean(predictedLinearSVC == X_test_targetSentiment)


# **Decision Tree Classifier**

# In[43]:


from sklearn.tree import DecisionTreeClassifier
clf_decisionTree_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), 
                                  ("clf_decisionTree", DecisionTreeClassifier())])
clf_decisionTree_pipe.fit(X_train, X_train_targetSentiment)

predictedDecisionTree = clf_decisionTree_pipe.predict(X_test)
np.mean(predictedDecisionTree == X_test_targetSentiment)


# **Random Forest Classifier**

# In[45]:


from sklearn.ensemble import RandomForestClassifier
clf_randomForest_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_randomForest", RandomForestClassifier())])
clf_randomForest_pipe.fit(X_train, X_train_targetSentiment)

predictedRandomForest = clf_randomForest_pipe.predict(X_test)
np.mean(predictedRandomForest == X_test_targetSentiment)





from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],    
             'tfidf__use_idf': (True, False), 
             } 
gs_clf_LinearSVC_pipe = GridSearchCV(clf_linearSVC_pipe, parameters, n_jobs=-1)
gs_clf_LinearSVC_pipe = gs_clf_LinearSVC_pipe.fit(X_train, X_train_targetSentiment)
new_text = ["The tablet is good, really liked it.", # positive
            "The tablet is ok, but it works fine.", # neutral
            "The tablet is not good, does not work very well."] # negative

X_train_targetSentiment[gs_clf_LinearSVC_pipe.predict(new_text)]


# In[48]:


predictedGS_clf_LinearSVC_pipe = gs_clf_LinearSVC_pipe.predict(X_test)
np.mean(predictedGS_clf_LinearSVC_pipe == X_test_targetSentiment)


# **Results:**
# - After testing some arbitrary reviews, it seems that our features is performing correctly with Positive, Neutral, Negative results
# - We also see that after running the grid search, our Support Vector Machine Classifier has improved to **94.08%** accuracy level

# ## 6.7 Detailed Performance Analysis of Support Vector Machine Classifier

# For detailed analysis, we will:
# - Analyze the best mean score of the grid search (classifier, parameters, CPU core)
# - Analyze the best estimator
# - Analyze the best parameter

# In[49]:


for performance_analysis in (gs_clf_LinearSVC_pipe.best_score_, 
                             gs_clf_LinearSVC_pipe.best_estimator_, 
                             gs_clf_LinearSVC_pipe.best_params_):
        print(performance_analysis)


# - Here we see that the best mean score of the grid search is 93.65% which is very close to our accuracy level of 94.08%
# - Our best estimator here is also displayed
# - Lastly, our best parameters are true for use_idf in tfidf, and ngram_range between 1,2

# In[50]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(X_test_targetSentiment, predictedGS_clf_LinearSVC_pipe))
print('Accuracy: {}'. format(accuracy_score(X_test_targetSentiment, predictedGS_clf_LinearSVC_pipe)))




from sklearn import metrics
metrics.confusion_matrix(X_test_targetSentiment, predictedGS_clf_LinearSVC_pipe)

