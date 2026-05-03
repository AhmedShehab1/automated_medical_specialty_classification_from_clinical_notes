# **Building the Logistic Regression Model**

In the logistic_reg.ipynb notebook, we took our raw medical text and built a machine learning pipeline to automatically categorize the notes into specific biological systems. Here is exactly how we did it:  

### 1. Data Cleaning & Scoping

When we first looked at the data, it was severely imbalanced across 40 different categories, many of which overlapped confusingly (like hospital billing codes). To ensure our model was mathematically valid and biologically meaningful, we filtered the dataset down to just four distinct target systems: Cardiovascular, Neurology, Obstetrics/Gynecology, and Gastroenterology.  

### 2. Translating Text to Math (TF-IDF)

AI algorithms cannot read English, so we used a technique called TF-IDF (Term Frequency-Inverse Document Frequency) to translate the clinical notes into a matrix of numbers. This method is highly effective because it mathematically penalizes generic words (like "patient" or "blood") and boosts the importance of rare, highly specific medical terminology.

### 3. Rigorous Training & Tuning

Instead of just training the model once, we used a Logistic Regression algorithm paired with 5-Fold Cross-Validation. This means the computer divided the data into five chunks, training on four and testing on the fifth, rotating until every chunk was tested. During this process, we also used GridSearchCV to automatically test different internal settings (hyperparameters)—like adjusting the class weights and the regularization strength (C)—to find the perfect balance and prevent the model from overfitting.

### 4. Final Evaluation

Once we found the optimal settings, we gave the model a "final exam" using the 20% of the data we had hidden away from the very beginning. The model achieved an impressive 88% overall accuracy. To fully understand its performance, we generated a detailed classification report (Precision, Recall, F1-Score) and plotted a visual Confusion Matrix to see exactly where the model excelled and where it occasionally mixed up the medical specialties.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# **Optimizing the Data: Dimensionality Reduction & Feature Selection**

After establishing our initial baseline using Logistic Regression, we recognized a potential challenge: our TF-IDF text translator was generating 500 distinct mathematical features (words and phrases) for every single clinical note.

Before moving on to more computationally demanding algorithms like Support Vector Machines (SVM), we ran a standalone experiment to see if we could "shrink" the dataset. The goal was to reduce the number of features to speed up training times and prevent overfitting, all without losing the crucial medical information needed to make accurate predictions.

We tested and compared three advanced dimensionality reduction techniques:

1. **Backward Elimination (Recursive Feature Elimination - RFE)**
Instead of mathematically scrambling the data, RFE acts like a pruning shear.

How it works: It starts with all 500 words, trains a model, and identifies the least useful words. It drops them, retrains, and repeats the process until only the most predictive words remain.

The Result: RFE aggressively narrowed the 500 features down to just the 50 most important medical terms (such as abdomen, bleeding, mri, and catheterization).

Performance: Amazingly, training a model on just these 50 carefully selected words actually improved our accuracy to nearly 96%. Furthermore, because the features are still real English words, this method remains highly interpretable for doctors and stakeholders.

2. **Principal Component Analysis (PCA)**

PCA takes a completely different, purely mathematical approach.

How it works: Instead of dropping words, PCA looks at how all 500 words interact and compresses them into a smaller set of abstract, uncorrelated mathematical "components" that capture the overall essence (variance) of the text.

The Result: We found that we could capture 95% of the dataset's total variance using only 298 components (a roughly 40% reduction in data size).

Performance: The PCA-transformed data maintained a strong accuracy of ~95%. However, because the data is now compressed into abstract numbers, we lose the ability to know which specific medical words are driving the predictions.

3. **Independent Component Analysis (ICA)**

How it works: Similar to PCA, ICA compresses the data into abstract components. However, instead of looking for variance, it specifically hunts for components that are statistically independent from one another.

**The Result & Performance:** Using the same 298 component limit for a fair comparison, ICA performed almost identically to PCA, achieving ~95% accuracy.

**Conclusion & Next Steps**

This experiment proved that our clinical text data can be heavily optimized. RFE proved to be the winner for this specific task, dropping 90% of the data volume while actually improving accuracy and maintaining total interpretability.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# **Support Vector Machine (SVM) Classification Models**

After establishing our baseline with Logistic Regression, we implemented a more powerful algorithm—Support Vector Machines (SVM)—to classify the clinical notes. We tested several versions of this model, iterating on the feature sets, target categories, and dimensionality reduction techniques to find the optimal setup.

### **Overview of Model Versions**

We developed two core pipelines for testing:

* **Model 1 (Basic Text):** This model relies only on the raw text from the doctor's transcription. It serves as our baseline to see how well the AI performs with no extra context.

* **Model 2 (Combined Text):** This model combines the transcription text, extracted keywords, and processed medical codes into one unified text block. This provides the AI with significantly more context to make its predictions.

To ensure our models were robust and not just "memorizing" the data (overfitting), all versions utilized GridSearchCV with K-Fold Cross-Validation. This allowed the computer to automatically test multiple hyperparameters (like the C penalty and kernel type) across different splits of the training data to find the best possible configuration.

**Phase 1: Focused Biological Systems (4 Categories)**
In the svm_model_v02.ipynb notebook, we focused our classification on four highly distinct biological systems: Cardiovascular, Neurology, Obstetrics/Gynecology, and Gastroenterology.

**Results:**

* **Model 1 (Basic Text):** Achieved an excellent 93% accuracy on the hidden test set. It proved highly capable of distinguishing between these four distinct systems using just the doctor's notes.

* **Model 2 (Combined Text):** Adding the keywords and medical codes slightly dropped the overall accuracy to 92%, but it achieved a near-perfect 100% precision on Obstetrics/Gynecology notes.

**Conclusion for Phase 1:** When dealing with a small number of distinct categories, the basic transcription text is sufficient for the SVM to achieve high accuracy. Providing extra features (Model 2) didn't offer a significant advantage and added unnecessary complexity.

**Phase 2: Expanded Classification (13 Categories)**
In the svm_model_v03.ipynb notebook, we significantly increased the difficulty of the task. We expanded the target categories from 4 up to 13 different medical specialties, including overlapping areas like Surgery, General Medicine, Radiology, and Consultations.

**Results:**

* **Model 1 (Basic Text):** The accuracy plummeted to 45%. The model struggled immensely, predicting "0" for almost half of the categories (meaning it couldn't reliably identify them at all). It became heavily biased toward the most common category (Surgery).

* **Model 2 (Combined Text):** The accuracy jumped significantly to 71%. While not perfect, providing the AI with the extracted keywords and medical codes gave it the necessary context to navigate the 13 complex, overlapping categories.

**Conclusion for Phase 2:** As the classification task becomes more complex and the categories begin to overlap in terminology, the AI requires the extra context provided by the Combined Text (Model 2) to maintain a reasonable level of accuracy.

**Phase 3: Dimensionality Reduction (PCA & RFE)**
To see if we could optimize the AI's processing power, we experimented with compressing the mathematical data before feeding it to the SVM. We tested this on the 4-category dataset.

* **Principal Component Analysis (PCA):** We used TruncatedSVD to compress the 10,000 text features down into 300 core mathematical "concepts." This resulted in a slight drop in accuracy (around 90% for both models). While it sped up the training process, we lost a small amount of predictive detail.

* **Recursive Feature Elimination (RFE):** We used RFE to aggressively eliminate the least useful words, forcing the AI to train on only the top 100 most important medical terms. This resulted in the lowest accuracy scores of the experiment (83% for Basic, 88% for Combined).

**Conclusion for Phase 3:** For this specific clinical text dataset, the SVM algorithm performs best when given access to the full, uncompressed vocabulary (using standard TF-IDF). Compressing the data via PCA or RFE resulted in a loss of valuable clinical nuances, lowering overall accuracy.


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# **Handling Class Imbalance with SMOTE & MCC**

In the `SMOTE.ipynb` notebook, we tackle one of the most common challenges in medical machine learning: **Severe Class Imbalance**. In real-world clinical data, common procedures (like Surgery) appear vastly more often than rare ones (like Hospice/Palliative Care), which can cause the AI to heavily bias its predictions.  

## The Problem: A Heavily Skewed Dataset

Upon inspecting our dataset of 4,966 clinical notes across 40 medical specialties, we discovered an extreme imbalance.  

* **Largest Class:** Surgery (1,088 samples).  
* **Smallest Class:** Hospice - Palliative Care (6 samples).  
* **Imbalance Ratio:** The largest class is over 181 times larger than the smallest class.  

## The Baseline & The "Accuracy Trap"

Before applying any fixes, we established a baseline by extracting 500 TF-IDF numerical features from our combined text (transcription + keywords) and training a standard **Random Forest Classifier**.  

This baseline model perfectly demonstrated the **Accuracy Trap**:  

* The model achieved an **Accuracy of ~25%**.  
* However, the classification report revealed that the model was **completely ignoring the minority classes**—scoring 0.00 in precision and recall for specialties like Allergy, Dentistry, and Dermatology.  
* Because "Surgery" makes up such a massive portion of the data, the AI learned it could simply guess the majority classes and still appear moderately "accurate," while failing entirely at the actual task of distinguishing rare specialties.

## The Evaluation Standard: MCC

Because standard Accuracy is highly misleading on imbalanced data, this pipeline utilizes the **Matthews Correlation Coefficient (MCC)**.  

Unlike Accuracy (which only looks at correct predictions) or F1-Score (which partially handles imbalance), MCC uses all four quadrants of the confusion matrix (True Positives, True Negatives, False Positives, False Negatives).  It outputs a score between -1 and +1.  

Our baseline model scored an MCC of only **0.1669**, proving that its predictive power was actually very close to random guessing despite the 25% accuracy. MCC is our gold-standard metric for evaluating success in this phase of the project.  

## The Solution: SMOTE (Synthetic Minority Over-sampling Technique)

To force the AI to learn the rare medical specialties, we implemented SMOTE from the `imbalanced-learn` library.
Instead of simply duplicating existing notes (which leads to memorization/overfitting), SMOTE looks at the mathematical features of the minority classes and generates entirely new, synthetic data points that blend the characteristics of the rare notes.  By balancing the training data mathematically before feeding it to the Random Forest model, we ensure the algorithm treats Hospice Care with the exact same priority and weight as Surgery, leading to a much smarter, fairer, and more robust clinical classifier. 


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
