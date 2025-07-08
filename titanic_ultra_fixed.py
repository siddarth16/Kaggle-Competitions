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

class UltraAdvancedTitanicPredictor:
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
    
    def ultra_feature_engineering(self, df):
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
        
        # Cabin features with advanced positioning
        data['Cabin_Known'] = data['Cabin'].notna().astype(int)
        data['Cabin_Deck'] = data['Cabin'].str.extract('([A-Za-z])', expand=False)
        data['Cabin_Deck'].fillna('Unknown', inplace=True)
        
        data['Cabin_Number'] = data['Cabin'].str.extract('(\d+)', expand=False)
        data['Cabin_Number'] = pd.to_numeric(data['Cabin_Number'], errors='coerce')
        data['Cabin_Number'].fillna(0, inplace=True)
        
        # Advanced cabin position based on historical Titanic layout
        def get_cabin_position_advanced(cabin_num, deck):
            if cabin_num == 0:
                return 'Unknown'
            
            if deck in ['A', 'B', 'C']:  # Upper decks
                if cabin_num < 20:
                    return 'Forward_Upper'
                elif cabin_num < 80:
                    return 'Amidships_Upper'
                else:
                    return 'Aft_Upper'
            elif deck in ['D', 'E']:  # Middle decks
                if cabin_num < 30:
                    return 'Forward_Middle'
                elif cabin_num < 90:
                    return 'Amidships_Middle'
                else:
                    return 'Aft_Middle'
            else:  # Lower decks
                if cabin_num < 40:
                    return 'Forward_Lower'
                elif cabin_num < 100:
                    return 'Amidships_Lower'
                else:
                    return 'Aft_Lower'
        
        data['Cabin_Position_Advanced'] = data.apply(
            lambda row: get_cabin_position_advanced(row['Cabin_Number'], row['Cabin_Deck']), axis=1
        )
        
        # Ship side
        data['Ship_Side'] = data['Cabin_Number'].apply(
            lambda x: 'Starboard' if x % 2 == 1 else 'Port' if x > 0 else 'Unknown'
        )
        
        # Distance to lifeboats (historical data)
        lifeboat_positions = {
            'A': [1, 3, 5, 7, 9, 11, 13, 15], 
            'B': [2, 4, 6, 8, 10, 12, 14, 16], 
            'C': [1, 3, 5, 7, 9, 11, 13, 15], 
            'D': [2, 4, 6, 8, 10, 12, 14, 16]
        }
        
        def distance_to_lifeboats(cabin_num, deck):
            if cabin_num == 0 or deck not in lifeboat_positions:
                return 999
            
            positions = lifeboat_positions.get(deck, [])
            if not positions:
                return 999
            
            return min([abs(cabin_num - pos*10) for pos in positions])
        
        data['Distance_to_Lifeboats'] = data.apply(
            lambda row: distance_to_lifeboats(row['Cabin_Number'], row['Cabin_Deck']), axis=1
        )
        data['Near_Lifeboats'] = (data['Distance_to_Lifeboats'] < 20).astype(int)
        
        # Advanced age features
        data['Age_Group'] = pd.cut(data['Age'], bins=[0, 1, 4, 12, 18, 35, 60, 100], 
                                 labels=['Infant', 'Toddler', 'Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
        
        # Age interactions
        data['Age_Class'] = data['Age'] * data['Pclass']
        data['Age_Sex'] = data['Age'] * data['Sex'].map({'male': 0, 'female': 1})
        data['Age_Fare'] = data['Age'] * data['Fare']
        
        # Fare features
        data['Fare_Per_Person'] = data['Fare'] / data['Family_Size']
        data['Fare_Bin'] = pd.qcut(data['Fare'], q=10, labels=False)
        
        # Advanced interaction features
        data['Sex_Pclass'] = data['Sex'] + '_' + data['Pclass'].astype(str)
        data['Embarked_Pclass'] = data['Embarked'] + '_' + data['Pclass'].astype(str)
        data['Title_Pclass'] = data['Title'] + '_' + data['Pclass'].astype(str)
        data['Age_Group_Sex'] = data['Age_Group'].astype(str) + '_' + data['Sex']
        
        # Statistical features by multiple groupings
        for groupby_col in ['Pclass', 'Sex', 'Embarked', 'Title']:
            for col in ['Age', 'Fare']:
                data[f'{col}_Mean_By_{groupby_col}'] = data.groupby(groupby_col)[col].transform('mean')
                data[f'{col}_Std_By_{groupby_col}'] = data.groupby(groupby_col)[col].transform('std')
                data[f'{col}_Median_By_{groupby_col}'] = data.groupby(groupby_col)[col].transform('median')
                data[f'{col}_Max_By_{groupby_col}'] = data.groupby(groupby_col)[col].transform('max')
                data[f'{col}_Min_By_{groupby_col}'] = data.groupby(groupby_col)[col].transform('min')
        
        # Survival patterns
        data['Women_Children_First'] = ((data['Sex'] == 'female') | (data['Age'] < 16)).astype(int)
        data['High_Class_Women'] = ((data['Sex'] == 'female') & (data['Pclass'] == 1)).astype(int)
        data['Third_Class_Male'] = ((data['Sex'] == 'male') & (data['Pclass'] == 3)).astype(int)
        data['Officer_Male'] = ((data['Title'] == 'Officer') & (data['Sex'] == 'male')).astype(int)
        
        # Polynomial features
        data['Age_Squared'] = data['Age'] ** 2
        data['Fare_Squared'] = data['Fare'] ** 2
        data['Age_Cubed'] = data['Age'] ** 3
        data['Fare_Cubed'] = data['Fare'] ** 3
        
        # Log transformations
        data['Age_Log'] = np.log1p(data['Age'])
        data['Fare_Log'] = np.log1p(data['Fare'])
        
        # Binning combinations
        data['Age_Fare_Bin'] = pd.cut(data['Age'] * data['Fare'], bins=20, labels=False)
        data['Family_Fare_Bin'] = pd.cut(data['Family_Size'] * data['Fare'], bins=20, labels=False)
        
        # Ticket features
        data['Ticket_Prefix'] = data['Ticket'].str.extract('([A-Za-z]+)', expand=False)
        data['Ticket_Prefix'].fillna('None', inplace=True)
        data['Ticket_Number'] = data['Ticket'].str.extract('(\d+)', expand=False)
        data['Ticket_Number'] = pd.to_numeric(data['Ticket_Number'], errors='coerce')
        data['Ticket_Number'].fillna(0, inplace=True)
        
        # Ticket prefix survival rates
        ticket_prefix_survival = data[train_mask].groupby('Ticket_Prefix')['Survived'].agg(['mean', 'count']).reset_index()
        ticket_prefix_survival.columns = ['Ticket_Prefix', 'Ticket_Prefix_Survival_Rate', 'Ticket_Prefix_Count']
        data = data.merge(ticket_prefix_survival, on='Ticket_Prefix', how='left')
        data['Ticket_Prefix_Survival_Rate'].fillna(0.5, inplace=True)
        data['Ticket_Prefix_Count'].fillna(1, inplace=True)
        
        return data
    
    def encode_features(self, data):
        categorical_features = ['Sex', 'Embarked', 'Title', 'Age_Group', 'Cabin_Deck', 
                              'Ticket_Prefix', 'Sex_Pclass', 'Embarked_Pclass', 'Title_Pclass',
                              'Family_Type', 'Cabin_Position_Advanced', 'Ship_Side', 'Age_Group_Sex']
        
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
    
    def pseudo_labeling(self, X_train, y_train, X_test, confidence_threshold=0.95):
        # Train initial models
        base_models = {
            'xgb': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED),
            'lgb': lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED),
            'rf': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_SEED)
        }
        
        # Get predictions from all models
        predictions = []
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_test)[:, 1]
            predictions.append(pred_proba)
        
        # Average predictions
        avg_predictions = np.mean(predictions, axis=0)
        
        # Select high-confidence predictions
        high_confidence_mask = (avg_predictions > confidence_threshold) | (avg_predictions < (1 - confidence_threshold))
        
        if high_confidence_mask.sum() > 0:
            # Add pseudo-labeled samples
            pseudo_labels = (avg_predictions > 0.5).astype(int)
            
            X_pseudo = X_test[high_confidence_mask]
            y_pseudo = pseudo_labels[high_confidence_mask]
            
            # Combine original and pseudo-labeled data
            X_combined = pd.concat([X_train, X_pseudo], ignore_index=True)
            y_combined = pd.concat([y_train, pd.Series(y_pseudo)], ignore_index=True)
            
            print(f"Added {len(X_pseudo)} pseudo-labeled samples")
            return X_combined, y_combined
        
        return X_train, y_train
    
    def train_ultra_ensemble(self, X_train, y_train, X_test):
        # Apply pseudo-labeling
        X_train_pseudo, y_train_pseudo = self.pseudo_labeling(X_train, y_train, X_test)
        
        # Define multiple model configurations
        models = {
            'xgb1': xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.05, 
                                    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED),
            'xgb2': xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.08, 
                                    subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED+1),
            'lgb1': lgb.LGBMClassifier(n_estimators=800, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED),
            'lgb2': lgb.LGBMClassifier(n_estimators=600, max_depth=8, learning_rate=0.08,
                                    subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED+1),
            'cat1': CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                                     random_state=RANDOM_SEED, verbose=False),
            'cat2': CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08,
                                     random_state=RANDOM_SEED+1, verbose=False),
            'rf1': RandomForestClassifier(n_estimators=800, max_depth=12, min_samples_split=2,
                                        min_samples_leaf=1, random_state=RANDOM_SEED),
            'rf2': RandomForestClassifier(n_estimators=600, max_depth=15, min_samples_split=3,
                                        min_samples_leaf=2, random_state=RANDOM_SEED+1),
            'et1': ExtraTreesClassifier(n_estimators=800, max_depth=12, min_samples_split=2,
                                      min_samples_leaf=1, random_state=RANDOM_SEED),
            'et2': ExtraTreesClassifier(n_estimators=600, max_depth=15, min_samples_split=3,
                                      min_samples_leaf=2, random_state=RANDOM_SEED+1),
            'gb': GradientBoostingClassifier(n_estimators=800, max_depth=6, learning_rate=0.05,
                                           subsample=0.8, random_state=RANDOM_SEED),
            'ada': AdaBoostClassifier(n_estimators=400, learning_rate=0.1, random_state=RANDOM_SEED),
            'svm': SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=RANDOM_SEED),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, 
                               learning_rate_init=0.01, random_state=RANDOM_SEED),
            'nb': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            'lda': LinearDiscriminantAnalysis(),
            'lr': LogisticRegression(C=1.0, random_state=RANDOM_SEED)
        }
        
        skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_SEED)
        
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
        
        # Simple but effective ensemble strategies
        strategies = {
            'mean': np.mean(test_predictions, axis=1),
            'median': np.median(test_predictions, axis=1),
            'weighted_mean': np.average(test_predictions, axis=1, weights=np.arange(1, test_predictions.shape[1]+1)),
            'rank_mean': np.mean([stats.rankdata(test_predictions[:, i]) for i in range(test_predictions.shape[1])], axis=0) / len(X_test)
        }
        
        # Meta-learner on out-of-fold predictions
        meta_model = LogisticRegression(C=1.0, random_state=RANDOM_SEED)
        meta_model.fit(oof_predictions, y_train)
        meta_predictions = meta_model.predict_proba(test_predictions)[:, 1]
        
        # Final ensemble
        final_predictions = (0.35 * meta_predictions + 
                           0.25 * strategies['weighted_mean'] + 
                           0.2 * strategies['mean'] + 
                           0.1 * strategies['rank_mean'] + 
                           0.1 * strategies['median'])
        
        return final_predictions
    
    def optimize_threshold(self, X_train, y_train):
        thresholds = np.arange(0.1, 0.9, 0.005)
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
    
    def run_ultra_pipeline(self):
        print("Starting Ultra-Advanced Titanic Prediction Pipeline...")
        
        combined_data = self.load_data()
        engineered_data = self.ultra_feature_engineering(combined_data)
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
        
        final_predictions = self.train_ultra_ensemble(X_train, y_train, X_test)
        
        # Optimize threshold
        best_threshold = self.optimize_threshold(X_train, y_train)
        print(f"Optimal threshold: {best_threshold:.4f}")
        
        binary_predictions = (final_predictions > best_threshold).astype(int)
        
        submission = pd.DataFrame({
            'PassengerId': pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId'],
            'Survived': binary_predictions
        })
        
        submission.to_csv('ultra_titanic_submission.csv', index=False)
        
        print(f"✅ Ultra submission saved as 'ultra_titanic_submission.csv'")
        print(f"📊 Predicted survival rate: {binary_predictions.mean():.2%}")
        print(f"📈 Average prediction confidence: {final_predictions.mean():.4f}")
        
        return submission

# Run the pipeline
if __name__ == "__main__":
    predictor = UltraAdvancedTitanicPredictor()
    submission = predictor.run_ultra_pipeline()
    print("\n🎉 ULTRA COMPLETE! Download 'ultra_titanic_submission.csv' from Kaggle output.") 