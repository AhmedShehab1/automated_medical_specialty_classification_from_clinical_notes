import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    df = pd.read_csv("processed_clinical_data.csv")
    
    # Display the first 5 rows to verify what the data looks like
    print("Previewing the first 5 rows of the dataset:")
    print(df[['medical_specialty', 'extracted_features']].head())
    print("\n")

    print("\n" + "="*50)
    print("CHECKING FOR MISSING DATA")
    print("="*50)
    # Check for missing values in the entire dataframe
    print("Missing values per column before cleaning:")
    print(df.isnull().sum())
    
    # Drop rows where we are missing the crucial target label or the features
    initial_row_count = len(df)
    df = df.dropna(subset=['extracted_features', 'medical_specialty'])
    print(f"\nDropped {initial_row_count - len(df)} rows due to missing essential data.")


    print("\n" + "="*50)
    print("BIOLOGICAL SYSTEM FILTERING (RESEARCH ALIGNED)")
    print("="*50)
    df['medical_specialty'] = df['medical_specialty'].str.strip()
    
    # We map the administrative labels to the 4 distinct biological systems 
    target_classes = [
        'Cardiovascular / Pulmonary', # Heart
        'Neurology',                  # Brain
        'Obstetrics / Gynecology',    # Reproductive
        'Gastroenterology'            # Digestive
    ]
    
    df = df[df['medical_specialty'].isin(target_classes)]
    
    print(f"Dataset refined to 4 biologically distinct systems.")
    print(f"Total patient notes remaining for training: {len(df)}")


    print("\n" + "="*50)
    print("PREPARING THE FEATURES (TEXT CLEANING)")
    print("="*50)
    # The features are currently saved as stringified lists: "['C123', 'C456']"
    # We use regex to remove the brackets, quotes, and commas so it becomes "C123 C456"
    df['clean_features'] = df['extracted_features'].str.replace(r"[\[\]',]", "", regex=True)
    
    X = df['clean_features'] # Inputs
    y = df['medical_specialty'] # Target Labels


    print("\n" + "="*50)
    print("VECTORIZATION (TEXT TO MATH)")
    print("="*50)
    # Translate the medical codes into a mathematical matrix using TF-IDF
    vectorizer = TfidfVectorizer()
    X_vectors = vectorizer.fit_transform(X)
    print(f"Created a mathematical matrix with shape: {X_vectors.shape}")
    print(f"(Rows = Notes, Columns = Unique Medical Codes found)")


    print("\n" + "="*50)
    print("TRAIN/TEST SPLIT")
    print("="*50)
    # Split the dataset: 80% to train the model, 20% to test its accuracy
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)
    print(f"Training on {X_train.shape[0]} notes.")
    print(f"Testing on {X_test.shape[0]} notes.")


    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*50)
    # Initialize the Logistic Regression model. 
    # class_weight='balanced' tells the algorithm to pay extra attention to the slightly smaller categories
    model = LogisticRegression(max_iter=1000, class_weight='balanced') 
    
    print("Model is learning the patterns...")
    model.fit(X_train, y_train)


    print("\n" + "="*50)
    print("EVALUATION AND RESULTS")
    print("="*50)
    # predecting the hidden 20% test data
    y_pred = model.predict(X_test)

    # Calculate overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n>>>> OVERALL ACCURACY: {accuracy * 100:.2f}% <<<<\n")
    
    # Print the detailed report card for each department
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()