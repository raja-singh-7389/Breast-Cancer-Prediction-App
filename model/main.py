import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


def create_model(data):
    # Separate features and target
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    
    return model, scaler


def get_clean_data():
    # Load and clean data
    data = pd.read_csv("data.csv")
    
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Convert target to binary values
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data


def main():
    # Get cleaned data
    data = get_clean_data()

    # Create model and scaler
    model, scaler = create_model(data)

    # Save model and scaler using joblib
    dump(model, 'model/model.joblib')
    dump(scaler, 'model/scaler.joblib')
    print("Model and scaler saved successfully!")


if __name__ == '__main__':
    main()
