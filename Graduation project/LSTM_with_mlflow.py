import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow 
import mlflow.tensorflow
import argparse
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential, regularizers
from Ready_data import X, y

import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

def lstm_model(X_train, y_train, X_test, y_test, plot_name, embedding_dim, max_len, vocab_size, epochs, batch_size):
    
    lstm_units=256
    dropout_rate=0.3
    mlflow.set_experiment(f'Reviews_detection')
    with mlflow.start_run() as run:
        mlflow.set_tag('clf', 'LSTM')
        # Log model parameters from manually set arguments
        mlflow.log_param("vocab_size", vocab_size)
        mlflow.log_param("embedding_size", embedding_dim)
        mlflow.log_param("LSTM_units", lstm_units)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_len", max_len)

        # Model creation
        '''model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification'''
        # Build the model
        model = Sequential()
        model.add(Embedding(input_dim=args.vocab_size, output_dim=args.embedding_dim, input_length=args.max_len))
        model.add(LSTM(256, recurrent_dropout=0.3, dropout=0.3))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Log model summary
        #model.summary(print_fn=lambda x: mlflow.log_text(f'{plot_name}_model_summary.txt'))

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

        # Log model
        mlflow.tensorflow.log_model(model, artifact_path=f'{model.__class__.__name__}/{plot_name}')

        # Predicting and calculating metrics
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype('int32')

        f1_test = f1_score(y_test, y_pred, average='micro')
        acc_test = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        print(f'Micro-averaged ROC AUC Score: {roc_auc}')
        print(f' accuracy Score: {acc_test}')

        # Log metrics
        mlflow.log_metrics({'accuracy': acc_test, 'f1-score': f1_test, 'auc': roc_auc})

        # Confusion Matrix
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        

        # Save Confusion Matrix to MLflow
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(figure=conf_matrix_fig, artifact_file= f'{plot_name}_conf_matrix.png')
        plt.close()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        

        # Save ROC curve to MLflow
        roc_fig = plt.gcf()
        mlflow.log_figure(figure=roc_fig, artifact_file= f'{plot_name}_roc_curve.png')
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        

        # Save Precision-Recall curve to MLflow
        pr_fig = plt.gcf()
        mlflow.log_figure(figure=pr_fig, artifact_file= f'{plot_name}_pr_curve.png')
        plt.close()

def main(embedding_dim: int, max_len: int, vocab_size: int, epochs: int, batch_size: int):

    # ---------------- Calling the above function -------------------- ##

    ## 1. without considering the imabalancing data
    lstm_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, plot_name='LSTM_first_try', 
            embedding_dim=embedding_dim, max_len=max_len, vocab_size=vocab_size, epochs=epochs, batch_size=batch_size)

if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', '-v', type=int, default=20000)
    parser.add_argument('--embedding_dim', '-ed', type=int, default=200)
    parser.add_argument('--max_len', '-m', type=int, default=150)
    parser.add_argument('--epochs', '-e', type=int, default=12)
    parser.add_argument('--batch_size', '-b', type=int, default=128)

    args = parser.parse_args()

    ## Call the main function
    main(embedding_dim=args.embedding_dim, max_len=args.max_len, vocab_size=args.vocab_size, epochs=args.epochs, batch_size=args.batch_size)






