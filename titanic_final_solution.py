import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class TitanicPredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        
    def load_data(self):
        train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
        test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
        
        self.train_len = len(train_df)
        self.test_len = len(test_df)
        
        train_df['is_train'] = 1
        test_df['is_train'] = 0
        test_df['Survived'] = -1
        
        self.combined_df = pd.concat([train_df, test_df], ignore_index=True)
        return self.combined_df
    
    def advanced_feature_engineering(self, df):
        data = df.copy()
        
        # Basic imputation
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        
        # Name features
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
            'Mlle': 'Miss', 'Countess': 'Mrs', 'Ms': 'Miss', 'Lady': 'Mrs',
            'Jonkheer': 'Officer', 'Don': 'Officer', 'Dona': 'Mrs', 'Mme': 'Mrs',
            'Capt': 'Officer', 'Sir': 'Officer'
        }
        data['Title'] = data['Title'].map(title_mapping)
        data['Title'].fillna('Other', inplace=True)
        
        data['Name_Length'] = data['Name'].str.len()
        data['Name_Word_Count'] = data['Name'].str.split().str.len()
        
        # Family features
        data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
        data['Is_Alone'] = (data['Family_Size'] == 1).astype(int)
        data['Family_Type'] = pd.cut(data['Family_Size'], bins=[0, 1, 4, 20], labels=['Alone', 'Small', 'Large'])
        
        # Advanced family survival features
        data['Last_Name'] = data['Name'].str.extract('([A-Za-z]+),', expand=False)
        
        # Ticket-based survival (very powerful feature)
        train_mask = data['is_train'] == 1
        ticket_survival = data[train_mask].groupby('Ticket')['Survived'].agg(['mean', 'count', 'sum']).reset_index()
        ticket_survival.columns = ['Ticket', 'Ticket_Survival_Rate', 'Ticket_Count', 'Ticket_Survivors']
        data = data.merge(ticket_survival, on='Ticket', how='left')
        data['Ticket_Survival_Rate'].fillna(0.5, inplace=True)
        data['Ticket_Count'].fillna(1, inplace=True)
        data['Ticket_Survivors'].fillna(0, inplace=True)
        
        # Sibling survival (same last name and SibSp > 0)
        sibling_survival = data[train_mask & (data['SibSp'] > 0)].groupby('Last_Name')['Survived'].agg(['mean', 'count']).reset_index()
        sibling_survival.columns = ['Last_Name', 'Sibling_Survival_Rate', 'Sibling_Count']
        data = data.merge(sibling_survival, on='Last_Name', how='left')
        data['Sibling_Survival_Rate'].fillna(0.5, inplace=True)
        data['Sibling_Count'].fillna(0, inplace=True)
        
        # Cabin features
        data['Cabin_Known'] = data['Cabin'].notna().astype(int)
        data['Cabin_Deck'] = data['Cabin'].str.extract('([A-Za-z])', expand=False)
        data['Cabin_Deck'].fillna('Unknown', inplace=True)
        
        data['Cabin_Number'] = data['Cabin'].str.extract('(\d+)', expand=False)
        data['Cabin_Number'] = pd.to_numeric(data['Cabin_Number'], errors='coerce')
        data['Cabin_Number'].fillna(0, inplace=True)
        
        # Cabin position
        def get_cabin_position(cabin_num):
            if cabin_num == 0:
                return 'Unknown'
            elif cabin_num < 30:
                return 'Forward'
            elif cabin_num < 80:
                return 'Amidships'
            else:
                return 'Aft'
        
        data['Cabin_Position'] = data['Cabin_Number'].apply(get_cabin_position)
        
        # Ship side
        data['Ship_Side'] = data['Cabin_Number'].apply(
            lambda x: 'Starboard' if x % 2 == 1 else 'Port' if x > 0 else 'Unknown'
        )
        
        # Distance to main staircases
        main_staircase_positions = [10, 20, 30, 50, 70, 90]
        data['Distance_to_Staircase'] = data['Cabin_Number'].apply(
            lambda x: min([abs(x - pos) for pos in main_staircase_positions]) if x > 0 else 999
        )
        data['Near_Staircase'] = (data['Distance_to_Staircase'] < 15).astype(int)
        
        # Advanced age imputation
        def impute_age_advanced(row):
            if pd.isna(row['Age']):
                if row['SibSp'] > 0 and row['Last_Name'] in data['Last_Name'].values:
                    sibling_ages = data[(data['Last_Name'] == row['Last_Name']) & 
                                      (data['SibSp'] > 0) & 
                                      (data['Age'].notna())]['Age']
                    if len(sibling_ages) > 0:
                        return sibling_ages.mean()
                
                class_sex_ages = data[(data['Pclass'] == row['Pclass']) & 
                                    (data['Sex'] == row['Sex']) & 
                                    (data['Age'].notna())]['Age']
                if len(class_sex_ages) > 0:
                    return class_sex_ages.mean()
                
                return data['Age'].median()
            return row['Age']
        
        data['Age'] = data.apply(impute_age_advanced, axis=1)
        
        # Age groups
        data['Age_Group'] = pd.cut(data['Age'], bins=[0, 1, 4, 12, 18, 35, 60, 100], 
                                 labels=['Infant', 'Toddler', 'Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
        
        # Fare features
        data['Fare_Per_Person'] = data['Fare'] / data['Family_Size']
        data['Fare_Bin'] = pd.qcut(data['Fare'], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
        
        # Interaction features
        data['Sex_Pclass'] = data['Sex'] + '_' + data['Pclass'].astype(str)
        data['Age_Class'] = data['Age'] * data['Pclass']
        data['Age_Sex'] = data['Age'] * data['Sex'].map({'male': 0, 'female': 1})
        data['Embarked_Pclass'] = data['Embarked'] + '_' + data['Pclass'].astype(str)
        data['Title_Pclass'] = data['Title'] + '_' + data['Pclass'].astype(str)
        
        # Statistical features
        for col in ['Age', 'Fare']:
            data[f'{col}_Mean_By_Pclass'] = data.groupby('Pclass')[col].transform('mean')
            data[f'{col}_Std_By_Pclass'] = data.groupby('Pclass')[col].transform('std')
            data[f'{col}_Median_By_Pclass'] = data.groupby('Pclass')[col].transform('median')
            data[f'{col}_Max_By_Pclass'] = data.groupby('Pclass')[col].transform('max')
            data[f'{col}_Min_By_Pclass'] = data.groupby('Pclass')[col].transform('min')
        
        # Survival patterns
        data['Women_Children_First'] = ((data['Sex'] == 'female') | (data['Age'] < 16)).astype(int)
        data['High_Class_Advantage'] = (data['Pclass'] == 1).astype(int)
        data['Male_Adult'] = ((data['Sex'] == 'male') & (data['Age'] >= 16)).astype(int)
        
        # Ticket features
        data['Ticket_Prefix'] = data['Ticket'].str.extract('([A-Za-z]+)', expand=False)
        data['Ticket_Prefix'].fillna('None', inplace=True)
        data['Ticket_Number'] = data['Ticket'].str.extract('(\d+)', expand=False)
        data['Ticket_Number'] = pd.to_numeric(data['Ticket_Number'], errors='coerce')
        data['Ticket_Number'].fillna(0, inplace=True)
        
        # Polynomial features
        data['Age_Squared'] = data['Age'] ** 2
        data['Fare_Squared'] = data['Fare'] ** 2
        data['Age_Fare_Interaction'] = data['Age'] * data['Fare']
        
        # Binning strategies
        data['Age_Fare_Bin'] = pd.cut(data['Age'] * data['Fare'], bins=10, labels=False)
        data['Family_Fare_Bin'] = pd.cut(data['Family_Size'] * data['Fare'], bins=10, labels=False)
        
        return data
    
    def encode_features(self, data):
        categorical_features = ['Sex', 'Embarked', 'Title', 'Age_Group', 'Cabin_Deck', 
                              'Ticket_Prefix', 'Fare_Bin', 'Sex_Pclass', 'Embarked_Pclass',
                              'Family_Type', 'Cabin_Position', 'Ship_Side', 'Title_Pclass']
        
        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                data[feature] = le.fit_transform(data[feature].astype(str))
                self.label_encoders[feature] = le
        
        return data
    
    def prepare_features(self, data):
        drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived', 'is_train', 'Last_Name']
        features = data.drop(columns=drop_features, errors='ignore')
        features = features.fillna(0)
        
        # Feature scaling
        scaler = RobustScaler()
        features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
        
        return features_scaled
    
    def train_ensemble_models(self, X_train, y_train, X_test):
        # Only models with predict_proba method
        models = {
            'xgb': xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05, 
                                   subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED),
            'lgb': lgb.LGBMClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED),
            'cat': CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05,
                                    random_state=RANDOM_SEED, verbose=False),
            'rf': RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=2,
                                       min_samples_leaf=1, random_state=RANDOM_SEED),
            'et': ExtraTreesClassifier(n_estimators=1000, max_depth=10, min_samples_split=2,
                                     min_samples_leaf=1, random_state=RANDOM_SEED),
            'gb': GradientBoostingClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05,
                                           subsample=0.8, random_state=RANDOM_SEED),
            'ada': AdaBoostClassifier(n_estimators=500, learning_rate=0.1, random_state=RANDOM_SEED),
            'svm': SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=RANDOM_SEED),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, 
                               learning_rate_init=0.01, random_state=RANDOM_SEED),
            'nb': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            'lda': LinearDiscriminantAnalysis(),
            'lr': LogisticRegression(C=1.0, random_state=RANDOM_SEED)
        }
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        
        oof_predictions = np.zeros((len(X_train), len(models)))
        test_predictions = np.zeros((len(X_test), len(models)))
        
        for i, (name, model) in enumerate(models.items()):
            print(f"Training {name}...")
            
            oof_preds = np.zeros(len(X_train))
            test_preds = np.zeros(len(X_test))
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
                test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
            
            oof_predictions[:, i] = oof_preds
            test_predictions[:, i] = test_preds
            
            cv_score = accuracy_score(y_train, (oof_preds > 0.5).astype(int))
            print(f"{name} CV Accuracy: {cv_score:.4f}")
        
        # Multi-level ensemble
        meta_models = {
            'lr_meta': LogisticRegression(C=1.0, random_state=RANDOM_SEED),
            'xgb_meta': xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED),
            'rf_meta': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
        }
        
        meta_predictions = np.zeros((len(X_test), len(meta_models)))
        
        for i, (name, meta_model) in enumerate(meta_models.items()):
            meta_model.fit(oof_predictions, y_train)
            meta_predictions[:, i] = meta_model.predict_proba(test_predictions)[:, 1]
        
        # Final ensemble strategies
        strategies = {
            'mean': np.mean(test_predictions, axis=1),
            'median': np.median(test_predictions, axis=1),
            'weighted_mean': np.average(test_predictions, axis=1, weights=np.arange(1, test_predictions.shape[1]+1)),
            'stacking': np.mean(meta_predictions, axis=1)
        }
        
        # Combined final prediction
        final_predictions = (0.4 * strategies['stacking'] + 
                           0.3 * strategies['weighted_mean'] + 
                           0.2 * strategies['mean'] + 
                           0.1 * strategies['median'])
        
        return final_predictions
    
    def optimize_threshold(self, X_train, y_train):
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        
        for threshold in thresholds:
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                temp_model = xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED)
                temp_model.fit(X_fold_train, y_fold_train)
                preds = temp_model.predict_proba(X_fold_val)[:, 1]
                
                score = accuracy_score(y_fold_val, (preds > threshold).astype(int))
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_threshold = threshold
        
        return best_threshold
    
    def run_complete_pipeline(self):
        print("Starting Advanced Titanic Prediction Pipeline...")
        
        combined_data = self.load_data()
        engineered_data = self.advanced_feature_engineering(combined_data)
        encoded_data = self.encode_features(engineered_data)
        features = self.prepare_features(encoded_data)
        
        train_mask = encoded_data['is_train'] == 1
        test_mask = encoded_data['is_train'] == 0
        
        X_train = features[train_mask]
        y_train = encoded_data[train_mask]['Survived']
        X_test = features[test_mask]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(X_train.columns)}")
        
        final_predictions = self.train_ensemble_models(X_train, y_train, X_test)
        
        # Optimize threshold
        best_threshold = self.optimize_threshold(X_train, y_train)
        print(f"Optimal threshold: {best_threshold:.4f}")
        
        binary_predictions = (final_predictions > best_threshold).astype(int)
        
        submission = pd.DataFrame({
            'PassengerId': pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId'],
            'Survived': binary_predictions
        })
        
        submission.to_csv('titanic_submission.csv', index=False)
        
        print(f"✅ Submission saved as 'titanic_submission.csv'")
        print(f"📊 Predicted survival rate: {binary_predictions.mean():.2%}")
        print(f"📈 Average prediction confidence: {final_predictions.mean():.4f}")
        
        return submission

# Run the pipeline
if __name__ == "__main__":
    predictor = TitanicPredictor()
    submission = predictor.run_complete_pipeline()
    print("\n🎉 COMPLETE! Download 'titanic_submission.csv' from Kaggle output.") 