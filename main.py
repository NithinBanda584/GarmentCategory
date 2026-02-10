from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import ttk
from tkinter import filedialog
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
     Dropout
)

from tensorflow.keras.applications.efficientnet import preprocess_input

import seaborn as sns
import os
import cv2
import joblib
import pickle
import warnings
import json
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from PIL import Image
import tensorflow as tf

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Activation, Dropout, Flatten,
    Conv2D, MaxPooling2D
)
from tensorflow.keras.utils import to_categorical

from keras.models import model_from_json
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from tensorflow.keras.models import load_model
from keras.preprocessing import image
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
#pip install -U efficientnet
#from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Choose EfficientNet variant (e.g., B0, B1, ..., B7)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

#base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # adjust input size accordingly
base_model.trainable = False  # Freeze base model


main = Tk('Hybrid Transfer Learning model for Automated Garment Category and Stiching Style Recognition')
main.geometry("1300x1200")


from PIL import Image, ImageTk

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bg_path = os.path.join(BASE_DIR, "backgorund.jpg")

print("BASE_DIR =", BASE_DIR)
print("FILES HERE =", os.listdir(BASE_DIR))

bg_image = Image.open(bg_path)


bg_image = Image.open(bg_path)
bg_image = bg_image.resize((1300, 1200))
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = Label(main, image=bg_photo)
bg_label.image = bg_photo
bg_label.place(x=0, y=0, relwidth=1, relheight=1)



global filename
global X, Y
global model
global categories,model_folder


model_folder = "model"

from tkinter import *

import os
from tkinter import filedialog, END

def uploadDataset():
    global filename, categories, all_classes,class_folders1
    text.delete('1.0', END)

    filename = filedialog.askdirectory(initialdir=".")
    
    # First-level folders like Cotton-Poly, Denim-Poly, etc.
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]

    # Sub-folders (classes inside each category)
    all_classes = {}
    for category in categories:
        category_path = os.path.join(filename, category)
        class_folders = [sub for sub in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, sub))]
        class_folders1 = [sub for sub in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, sub))]

        all_classes[category] = class_folders

    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Top-level categories found: " + str(categories) + "\n")
    for cat in all_classes:
        text.insert(END, f"{cat} has {len(all_classes[cat])} classes: {all_classes[cat]}\n")

    
def imageProcessing():
    text.delete('1.0', END)
    global X, Y1, Y2, model_folder, filename, X_file, Y1_file, Y2_file,categories,base_model
    path = filename
    model_folder = "model"

    # Create model folder if not exists
    os.makedirs(model_folder, exist_ok=True)

    # Output file paths
    X_file = os.path.join(model_folder, "X_compressed.npz")
    Y1_file = os.path.join(model_folder, "Y1.npy")
    Y2_file = os.path.join(model_folder, "Y2.npy")
    category_map_file = os.path.join(model_folder, "categories.json")
    subclass_map_file = os.path.join(model_folder, "subclasses.json")

    # Build category and subclass maps
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    categories.sort()
    category_to_index = {cat: i for i, cat in enumerate(categories)}
    subclass_to_index = {}

    for category in categories:
        sub_path = os.path.join(path, category)
        subfolders = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
        subfolders.sort()
        subclass_to_index[category] = {sub: i for i, sub in enumerate(subfolders)}

    # Load if already processed
    if os.path.exists(X_file) and os.path.exists(Y1_file) and os.path.exists(Y2_file):
        X = np.load(X_file)['X']
        Y1 = np.load(Y1_file)
        Y2 = np.load(Y2_file)
        print("X, Y1, Y2 loaded successfully from cache.")
    else:
        X = []
        Y1 = []
        Y2 = []

        for category in categories:
            cat_path = os.path.join(path, category)
            subfolders = subclass_to_index[category].keys()

            for subclass in subfolders:
                subclass_path = os.path.join(cat_path, subclass)
                for img_file in os.listdir(subclass_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'Thumbs.db' not in img_file:
                        img_path = os.path.join(subclass_path, img_file)
                        img = image.load_img(img_path, target_size=(128, 128))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)  
                        features = base_model.predict(x)
                        features = np.squeeze(features)
                        X.append(features)
                        Y1.append(category_to_index[category])
                        Y2.append(subclass_to_index[category][subclass])

        X = np.array(X)
        Y1 = np.array(Y1)
        Y2 = np.array(Y2)

        # Save data
        np.savez_compressed(X_file, X=X)
        np.save(Y1_file, Y1)
        np.save(Y2_file, Y2)

        with open(category_map_file, 'w') as f:
            json.dump(category_to_index, f)

        with open(subclass_map_file, 'w') as f:
            json.dump(subclass_to_index, f)

    text.insert(END, "Dataset Preprocessing completed\n")

def Train_Test_split():
    global X, Y1, Y2, x_train, x_test, y1_train, y1_test, y2_train, y2_test

    # Perform train-test split for all three arrays together
    x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, Y1, Y2, test_size=0.20, random_state=42
    )

    text.insert(END, "Total samples found in training dataset: " + str(x_train.shape) + "\n")
    text.insert(END, "Total samples found in testing dataset: " + str(x_test.shape) + "\n")
    text.insert(END, "Total samples found in y1 trainings dataset: " + str(y1_train.shape) + "\n")
    text.insert(END, "Total samples found in y2 trainings dataset: " + str(y2_train.shape) + "\n")

def calculateMetrics(algorithm,categories, predict, y_test):

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()  

    n_classes = len(categories)

    # Binarize true labels and predictions
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    y_pred_bin = label_binarize(predict, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(7, 6))
    colors = cycle(['red', 'blue', 'green', 'orange', 'purple', 'brown'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"Class {categories[i]} (AUC = {roc_auc[i]:.2f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(algorithm + " - ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()     



from sklearn.linear_model import Perceptron
from sklearn.multioutput import MultiOutputClassifier


from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.models import load_model
from keras.callbacks import EarlyStopping
import os

def Existing_DNN():
    global x_train, x_test, y1_train, y1_test, y2_train, y2_test, model_folder, categories, class_folders1
    text.delete('1.0', END)

    model_file = os.path.join(model_folder, "DNN_multioutput_model.h5")

    # Combine outputs
    y_train_combined = [y1_train, y2_train]
    y_test_combined = [y1_test, y2_test]

    # Flatten input
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))
    input_shape = x_train_flat.shape[1]

    num_classes_y1 = len(np.unique(y1_train))
    num_classes_y2 = len(np.unique(y2_train))

    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        inputs = Input(shape=(input_shape,))

        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.7)(x)
        x = Dense(64, activation='relu')(x)

        # Output 1
        y1_out = Dense(num_classes_y1, activation='softmax', name='y1_output')(x)

        # Output 2
        y2_out = Dense(num_classes_y2, activation='softmax', name='y2_output')(x)

        model = Model(inputs=inputs, outputs=[y1_out, y2_out])
        model.compile(
    optimizer='adam',
    loss={
        'y1_output': 'sparse_categorical_crossentropy',
        'y2_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'y1_output': 'accuracy',
        'y2_output': 'accuracy'
    }
)


        model.fit(x_train_flat,
                  y_train_combined,
                  epochs=3,
                  batch_size=64,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=2)],
                  verbose=1)

        model.save(model_file)
        print(f"Saved DNN multi-output model to {model_file}")

    # Predict
    y1_pred_probs, y2_pred_probs = model.predict(x_test_flat)
    y1_pred = np.argmax(y1_pred_probs, axis=1)
    y2_pred = np.argmax(y2_pred_probs, axis=1)

    # Evaluate
    calculateMetrics("DNN MultiOutput - Category (Y1)", categories, y1_pred, y1_test)
    calculateMetrics("DNN MultiOutput - Subclass (Y2)", class_folders1, y2_pred, y2_test)


def Existing_Perceptron():
    global x_train, x_test, y1_train, y1_test, y2_train, y2_test, model_folder, categories, class_folders1
    text.delete('1.0', END)

    model_file = os.path.join(model_folder, "Perceptron_multioutput_model.pkl")

    # Combine targets
    y_train_combined = np.column_stack((y1_train, y2_train))
    y_test_combined = np.column_stack((y1_test, y2_test))

    # Flatten the input shape (samples, 4, 4, 1280) → (samples, 20480)
    x_train_reshaped = x_train.reshape((x_train.shape[0], -1))
    x_test_reshaped = x_test.reshape((x_test.shape[0], -1))

    # Load or train model
    if os.path.exists(model_file):
        model = joblib.load(model_file)
    else:
        base_perceptron = Perceptron(
            max_iter=1,          # very quick training
            tol=None,            # disables early stopping
            shuffle=True,
            eta0=0.6       # small learning rate
            )
        model = MultiOutputClassifier(base_perceptron)
        model.fit(x_train_reshaped, y_train_combined)
        joblib.dump(model, model_file, compress=5)
        print(f'Multi-output Perceptron model saved to {model_file}')

    # Predict and evaluate
    y_pred_combined = model.predict(x_test_reshaped)
    y1_pred = y_pred_combined[:, 0]
    y2_pred = y_pred_combined[:, 1]

    calculateMetrics("Perceptron MultiOutput - Category (Y1)", categories, y1_pred, y1_test)
    calculateMetrics("Perceptron MultiOutput - Subclass (Y2)", class_folders1, y2_pred, y2_test)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

def Proposed_MLP():
    global x_train, x_test, y1_train, y1_test, y2_train, y2_test, model_folder, categories, class_folders1
    text.delete('1.0', END)

    model_file = os.path.join(model_folder, "MLP_multioutput_model.pkl")

    # Prepare combined targets
    y_train_combined = np.column_stack((y1_train, y2_train))
    y_test_combined = np.column_stack((y1_test, y2_test))

    # Reshape x data: (samples, 4, 4, 1280) → (samples, 4*4*1280)
    x_train_reshaped = x_train.reshape((x_train.shape[0], -1))
    x_test_reshaped = x_test.reshape((x_test.shape[0], -1))

    # Load or train model
    if os.path.exists(model_file):
        model = joblib.load(model_file)
    else:
        base_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',     
        solver='sgd',              
        learning_rate_init=1e-3,  
        max_iter=20,                
        batch_size=64,           
        random_state=42)
        model = MultiOutputClassifier(base_mlp)
        model.fit(x_train_reshaped, y_train_combined)
        joblib.dump(model, model_file, compress=5)
        print(f'Multi-output MLP model saved to {model_file}')

    # Predict both Y1 and Y2
    y_pred_combined = model.predict(x_test_reshaped)
    y1_pred = y_pred_combined[:, 0]
    y2_pred = y_pred_combined[:, 1]

    # Evaluate
    calculateMetrics("MLP MultiOutput - Category (Y1)", categories, y1_pred, y1_test)
    calculateMetrics("MLP MultiOutput - Subclass (Y2)", class_folders1, y2_pred, y2_test)


from sklearn.linear_model import Perceptron

import json

with open(os.path.join(model_folder, "categories.json"), "r") as f:
    category_to_index = json.load(f)

with open(os.path.join(model_folder, "subclasses.json"), "r") as f:
    subclass_to_index = json.load(f)

index_to_category = {v: k for k, v in category_to_index.items()}
index_to_subclass = {}

for cat in subclass_to_index:
    index_to_subclass[cat] = {v: k for k, v in subclass_to_index[cat].items()}


def predictMLP():
    global model_folder, categories, class_folders1, base_model

    text.delete('1.0', END)

    model_file = os.path.join(model_folder, "MLP_multioutput_model.pkl")

    if not os.path.exists(model_file):
        text.insert(END, "Model not found. Please train the model first.\n")
        return

    # Load trained model
    model = joblib.load(model_file)

    # Select image for prediction
    filename = filedialog.askopenfilename(initialdir="testImages")
    if not filename:
        text.insert(END, "No file selected.\n")
        return

    # Load and preprocess image
    img = image.load_img(filename, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) 

    features = base_model.predict(x)
    features = np.squeeze(features)
    x_input = features.reshape(1, -1)

    # Predict
    y_pred_combined = model.predict(x_input)
    y1_pred = y_pred_combined[0][0]
    y2_pred = y_pred_combined[0][1]

    # Get class names
    category_name = index_to_category.get(y1_pred, "Unknown")
    subclasses = list(index_to_subclass[category_name].values())
    subclass_name = subclasses[y2_pred] if y2_pred < len(subclasses) else "Unknown"



    # Display prediction in text box
    result = f"Predicted Category (Y1): {category_name}\nPredicted Subclass (Y2): {subclass_name}"
    text.insert(END, result + "\n")

    # Load original image for display
    disp_img = cv2.imread(filename)
    disp_img = cv2.resize(disp_img, (500, 500))

    # Overlay prediction text
    label1 = f"Cloth type: {category_name}"
    label2 = f"Stitching type: {subclass_name}"
    cv2.putText(disp_img, label1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(disp_img, label2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show image with predictions
    cv2.imshow("Prediction", disp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


from tinydb import TinyDB, Query
import hashlib

db = TinyDB("users_db.json")
users_table = db.table("users")


def signup(role):
    def register():
        u, p = user.get(), pwd.get()
        if not u or not p:
            messagebox.showerror("Error", "Fill all fields")
            return

        hashed = hashlib.sha256(p.encode()).hexdigest()
        User = Query()

        if users_table.search((User.username == u) & (User.role == role)):
            messagebox.showerror("Error", "User already exists")
            return

        users_table.insert({
            "username": u,
            "password": hashed,
            "role": role
        })
        messagebox.showinfo("Success", "Signup successful")
        win.destroy()

    win = Toplevel(main)
    win.title("Signup")
    win.geometry("300x200")
    Label(win, text="Username").pack(pady=5)
    user = Entry(win); user.pack()
    Label(win, text="Password").pack(pady=5)
    pwd = Entry(win, show="*"); pwd.pack()
    Button(win, text="Signup", command=register).pack(pady=10)

def login(role):
    def verify():
        u, p = user.get(), pwd.get()
        hashed = hashlib.sha256(p.encode()).hexdigest()
        User = Query()

        if users_table.search(
            (User.username == u) &
            (User.password == hashed) &
            (User.role == role)
        ):
            messagebox.showinfo("Success", "Login successful")
            win.destroy()
            clear_buttons()
            show_admin_buttons() if role == "Admin" else show_user_buttons()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    win = Toplevel(main)
    win.title("Login")
    win.geometry("300x200")
    Label(win, text="Username").pack(pady=5)
    user = Entry(win); user.pack()
    Label(win, text="Password").pack(pady=5)
    pwd = Entry(win, show="*"); pwd.pack()
    Button(win, text="Login", command=verify).pack(pady=10)

def clear_buttons():
    for w in main.winfo_children():
        if isinstance(w, Button):
            w.destroy()

    if 'bg_label' in globals():
        bg_label.lower()

    if 'title' in globals():
        title.lift()

    if 'text' in globals():
        text.lift()


def show_admin_buttons():
    clear_buttons()
    font = ('times', 13, 'bold')

    Button(main, text="Upload Dataset", command=uploadDataset, font=font, bg = 'azure', fg = 'black').place(x=20, y=150)
    Button(main, text="FE with EfficientNet", command=imageProcessing, font=font, bg = 'oldlace', fg = 'black').place(x=20, y=200)
    Button(main, text="Train Test Split", command=Train_Test_split, font=font, bg = 'cornsilk', fg = 'black').place(x=20, y=250)
    Button(main, text="Train DNN", command=Existing_DNN, font=font, bg = 'lightcyan', fg = 'black').place(x=20, y=300)
    Button(main, text="Train Perceptron", command=Existing_Perceptron, font=font, bg = 'whitesmoke', fg = 'black').place(x=20, y=350)
    Button(main, text="Train MLP", command=Proposed_MLP, font=font, bg = 'beige', fg = 'black').place(x=20, y=400)

    Button(main, text="Logout", bg="red",
           command=show_login_screen, font=font).place(x=20, y=450)

def show_user_buttons():
    clear_buttons()
    font = ('times', 13, 'bold')

    Button(main, text="Predict Garment Image",
           command=predictMLP, font=font, bg = 'orange', fg = 'black').place(x=20, y=300)

    Button(main, text="Logout", bg="honeydew",
           command=show_login_screen, font=font).place(x=20, y=350)

def show_login_screen():
    clear_buttons()
    font = ('times', 14, 'bold')

    # ADMIN BUTTONS (RED)
    Button(main, text="Admin Signup",
           command=lambda: signup("Admin"),
           bg="#2AC82A", fg="white",
           font=font).place(x=200, y=100)

    Button(main, text="Admin Login",
           command=lambda: login("Admin"),
           bg="#2AC82A", fg="white",
           font=font).place(x=700, y=100)

    # USER BUTTONS (GREEN / BLUE)
    Button(main, text="User Signup",
           command=lambda: signup("User"),
           bg="#2AC82A", fg="white",
           font=font).place(x=450, y=100)

    Button(main, text="User Login",
           command=lambda: login("User"),
           bg="#43BD43", fg="white",
           font=font).place(x=950, y=100)

title = Label(
    main,
    text="Hybrid Transfer Learning model for Automated Garment Category and Stiching Style Recognition",
    font=('times', 20, 'bold'),
    bg='lightblue'
)
title.place(x=250, y=10)
text = Text(main, height=25, width=80, font=('times', 12, 'bold'))
text.place(x=300, y=200)

show_login_screen()
main.mainloop()

