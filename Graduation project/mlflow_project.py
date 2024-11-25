import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import mlflow
import xgboost as xgb
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from test import transformed_reviews_df, y

import warnings
warnings.filterwarnings('ignore')

#path1 ='filename.csv'
#df=pd.read_csv(path1)

#path2 ='modified_dataset.csv'
#X=pd.read_csv(path2)

#df = df.sample(frac=0.5, random_state=42)


#df['Score']=df['Score']-1

#print(df.dtypes())
#print(df.head())

#tf=TfidfVectorizer()
#x=tf.fit_transform(df['cleaned_text'])

#transformed_reviews_df=pd.DataFrame(x.toarray(),columns=tf.get_feature_names_out())

#print(transformed_reviews_df.head())


#X=df['cleaned_text']


X_train, X_test, y_train, y_test = train_test_split(transformed_reviews_df, y, test_size=0.2, shuffle=True, random_state=0)

'''
from sklearn.decomposition import PCA

pca = PCA(n_components=10)  # Reduce to 10 principal components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

'''

## --------------------- Modeling ---------------------------- ##



def xgb_model(X_train, y_train, X_test, y_test, plot_name, n_estimators, max_depth):



    mlflow.set_experiment(f'detection_with_xgb')
    with mlflow.start_run() as run:
        mlflow.set_tag('clf', 'xgboost')

        # Try XGBoost classifier
        clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                random_state=45,  tree_method='hist')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        ## metrics
        f1_test = f1_score(y_test, y_pred, average='micro')
        acc_test = accuracy_score(y_test, y_pred)
        acc_train = clf.score(X_train,y_train)
        roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        # Print or log the micro-averaged ROC AUC score
        print(f'Micro-averaged ROC AUC Score: {roc_auc}')

        # Log params, metrics, and model 
        mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
        mlflow.log_metrics({'train_accuracy': acc_train,'accuracy': acc_test, 'f1-score': f1_test,'auc': roc_auc} )
        mlflow.sklearn.log_model(clf, f'{clf.__class__.__name__}/{plot_name}')

        ## Plot the confusion matrix and save it to mlflow
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cbar=False, fmt='.2f', cmap='Blues')
        plt.title(f'{plot_name}')
        plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
        plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

        # Save to MLflow
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(figure=conf_matrix_fig, artifact_file=f'{plot_name}_conf_matrix.png')
        plt.close()

        # Compute ROC curve and AUC
        #fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
        #roc_auc = auc(fpr, tpr)

        #roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr', average='micro')

        # Print or log the micro-averaged ROC AUC score
        print(f'Micro-averaged ROC AUC Score: {roc_auc}')

        # Plot ROC curve for micro-average (if desired, display it in the title)
        plt.figure(figsize=(10, 8))

        # You can plot the ROC AUC score in the title, as no curve is drawn directly for micro-average
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Micro-Averaged ROC AUC = {roc_auc:.2f}')

        # Save the plot to MLflow
        roc_fig = plt.gcf()
        mlflow.log_figure(figure=roc_fig, artifact_file=f'{plot_name}_roc_curve_micro.png')
        plt.close()

        

def svc_model(X_train, y_train, X_test, y_test, plot_name, kernel, C, gamma):

    mlflow.set_experiment(f'detection_with_svc')
    with mlflow.start_run() as run:
        mlflow.set_tag('clf', 'svc')

        # Try SVC classifier
        clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=45)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)


        ## metrics
        f1_test = f1_score(y_test, y_pred, average='micro')
        acc_test = clf.score(X_test,y_test)
        acc_train = clf.score(X_train,y_train)
        roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        # Print or log the micro-averaged ROC AUC score
        print(f'Micro-averaged ROC AUC Score: {roc_auc}')
        
        # Log params, metrics, and model 
        mlflow.log_params({'kernel': kernel, 'C': C, 'gamma': gamma})
        mlflow.log_metrics({'train_accuracy': acc_train, 'accuracy': acc_test, 'f1-score': f1_test, 'auc': roc_auc})
        mlflow.sklearn.log_model(clf, f'{clf.__class__.__name__}/{plot_name}')

        ## Plot the confusion matrix and save it to mlflow
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cbar=False, fmt='.2f', cmap='Blues')
        plt.title(f'{plot_name}')
        plt.xticks(ticks=np.arange(len(np.unique(y_test))) + 0.5, labels=np.unique(y_test))
        plt.yticks(ticks=np.arange(len(np.unique(y_test))) + 0.5, labels=np.unique(y_test))

        
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        

        # Save to MLflow
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(figure=conf_matrix_fig, artifact_file=f'{plot_name}_conf_matrix.png')
        plt.close()

        # Compute ROC curve and AUC
        #roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr', average='micro')

        # Print or log the micro-averaged ROC AUC score
        print(f'Micro-averaged ROC AUC Score: {roc_auc}')

        # Plot ROC curve for micro-average (if desired, display it in the title)
        plt.figure(figsize=(10, 8))

        # You can plot the ROC AUC score in the title, as no curve is drawn directly for micro-average
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Micro-Averaged ROC AUC = {roc_auc:.2f}')

    '''fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test))
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # random guessing
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")'''
        

        # Save the plot to MLflow
    '''roc_fig = plt.gcf()
        mlflow.log_figure(figure=roc_fig, artifact_file=f'{plot_name}_roc_curve_micro.png')
        plt.close()'''

        #precision, recall, _ = precision_recall_curve(y_test, clf.predict_proba(X_test))
        
    '''plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")'''
        

        # Save the plot to MLflow
    '''roc_fig = plt.gcf()
        mlflow.log_figure(figure=roc_fig, artifact_file=f'{plot_name}_roc_curve_micro.png')
        plt.close()'''


def logistic_regression_model(X_train, y_train, X_test, y_test, plot_name, C, solver, max_iter):

    mlflow.set_experiment(f'detection_with_logistic_regression')
    with mlflow.start_run() as run:
        mlflow.set_tag('clf', 'logistic_regression')

        # Instantiate and train Logistic Regression model
        clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter, multi_class='ovr', random_state=45)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        ## metrics
        f1_test = f1_score(y_test, y_pred, average='micro')
        acc_test = accuracy_score(y_test, y_pred)
        acc_train = clf.score(X_train,y_train)
        roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    
        # Print or log the micro-averaged ROC AUC score
        print(f'Micro-averaged ROC AUC Score: {roc_auc}')

        # Log params, metrics, and model 
        mlflow.log_params({'C': C, 'solver': solver, 'max_iter': max_iter})
        mlflow.log_metrics({'train_accuracy': acc_train, 'accuracy': acc_test, 'f1-score': f1_test, 'auc': roc_auc})
        mlflow.sklearn.log_model(clf, f'{clf.__class__.__name__}/{plot_name}')

       
        
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        

        # Save to MLflow
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(figure=conf_matrix_fig, artifact_file=f'{plot_name}_conf_matrix.png')
        plt.close()

        # Compute ROC curve and AUC
        #roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr', average='micro')

        # Print or log the micro-averaged ROC AUC score
        print(f'Micro-averaged ROC AUC Score: {roc_auc}')

        # Plot ROC curve for micro-average (if desired, display it in the title)
        plt.figure(figsize=(10, 8))

        # You can plot the ROC AUC score in the title, as no curve is drawn directly for micro-average
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Micro-Averaged ROC AUC = {roc_auc:.2f}')

        '''fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test))
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # random guessing
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        

        # Save the plot to MLflow
        roc_fig = plt.gcf()
        mlflow.log_figure(figure=roc_fig, artifact_file=f'{plot_name}_roc_curve_micro.png')
        plt.close()'''

        '''precision, recall, _ = precision_recall_curve(y_test, clf.predict_proba(X_test))
        
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        

        # Save the plot to MLflow
        roc_fig = plt.gcf()
        mlflow.log_figure(figure=roc_fig, artifact_file=f'{plot_name}_roc_curve_micro.png')
        plt.close()'''





def main(n_estimators: int, max_depth: int, cs: float, cl: float, max_iter:int):

    # ---------------- Calling the above function -------------------- ##

    ## 1. without considering the imabalancing data
    '''xgb_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, plot_name='XG_first_try', 
                n_estimators=n_estimators, max_depth=max_depth)'''
    
    svc_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, plot_name='svc_first_try', 
                kernel='rbf', C=cs, gamma='scale')
        # kernals to try (poly , rbf)

    '''logistic_regression_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, plot_name='lr_first_try',
                               C=cl, solver='lbfgs', max_iter=100)'''





if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--max_depth', '-d', type=int, default=15)
    parser.add_argument('--c_svc', '-cs', type=float, default=1.0)
    parser.add_argument('--c_lr', '-cl', type=float, default=1.0)
    parser.add_argument('--max_iter', '-i', type=int, default=100)

    args = parser.parse_args()

    ## Call the main function
    main(n_estimators=args.n_estimators, max_depth=args.max_depth, cs=args.c_svc, cl=args.c_lr, max_iter=args.max_iter)

