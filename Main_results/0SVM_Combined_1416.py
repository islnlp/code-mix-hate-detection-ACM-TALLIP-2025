import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Load the original dataset and additional dataset into pandas DataFrames
original_data = pd.read_csv("/data1/aakash/Codemix/Aakash_02/1HateSpeech_Codemix.csv")
original_data=original_data.dropna()
eng = pd.read_csv("/data1/aakash/Codemix/Aakash_02/1Hatespeech_English(new).csv")
hind = pd.read_csv("/data1/aakash/Codemix/Aakash_02/1HateSpeech_Hindi.csv")


# Select rows with the specified label
da = eng[eng['Tag'] == 1].iloc[:1416]
da = da.reset_index(drop=True)

db = eng[eng['Tag']==0].iloc[:1416]
db = db.reset_index(drop=True)

english_data= pd.concat([da, db])#train_dx
english_data = english_data.reset_index(drop=True)
english_data


# Select rows with the specified label
dk = hind[hind['Tag'] == 1].iloc[:1416]
dk = dk.reset_index(drop=True)

dm = hind[hind['Tag']==0].iloc[:1416]
dm = dm.reset_index(drop=True)

hindi_data= pd.concat([dk, dm])#train_dy
hindi_data = hindi_data.reset_index(drop=True)
hindi_data


# Step 2: Split the original dataset into train, test, and validation sets
X_original = original_data['Sentence']
y_original = original_data['Tag']

X_train_original, X_remaining_original, y_train_original, y_remaining_original = train_test_split(X_original, y_original, test_size=0.3, random_state=42, stratify=y_original)
X_test_original, X_val_original, y_test_original, y_val_original = train_test_split(X_remaining_original, y_remaining_original, test_size=0.5, random_state=42, stratify=y_remaining_original)


# Step 3: Combine the train set from the original dataset with the additional dataset
combined_train = pd.concat([X_train_original, english_data['Sentence'], hindi_data['Sentence']])
combined_train_labels = pd.concat([y_train_original, english_data['Tag'], hindi_data['Tag']])


# Step 4: Preprocess the combined train set and additional test and validation sets (extract n-gram features)
ngram_vectorizer = CountVectorizer(ngram_range=(1, 3))
X_train_combined = ngram_vectorizer.fit_transform(combined_train)
X_test_original_ngrams = ngram_vectorizer.transform(X_test_original)
X_val_original_ngrams = ngram_vectorizer.transform(X_val_original)


# Step 5: Train the classifier on the combined train set
svm_classifier = SVC(kernel='linear')#RandomForestClassifier(n_estimators=100, random_state=42)#MultinomialNB()#SVC(kernel='linear')
svm_classifier.fit(X_train_combined, combined_train_labels)


# Step 6: Evaluate the classifier on the test and validation sets from the original dataset
y_pred_test_original = svm_classifier.predict(X_test_original_ngrams)
y_pred_val_original = svm_classifier.predict(X_val_original_ngrams)

accuracy_test_original = accuracy_score(y_test_original, y_pred_test_original)
accuracy_val_original = accuracy_score(y_val_original, y_pred_val_original)

print("Test Set Accuracy on Original Data:", accuracy_test_original)
print("Validation Set Accuracy on Original Data:", accuracy_val_original)


# Step 7: Print the classification report for the test set on the combined dataset
print("\nClassification Report for Test Set on Combined Dataset:")
print(classification_report(y_test_original, y_pred_test_original))

