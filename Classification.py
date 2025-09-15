"""
Fichier de brouillon pour tester la récupération des données et l'association aux annotations. Entrainement d'un Random Forest sur 80% des données (train) et test sur 20% (test), puis test du seuil AC>75 en combinant les annotations après calculs des AC à 1s.

1. Lire les fichiers BIN, conversion dans la bonne unité et transformer en AC
2. Récuperer les annotations
3. Associer annotations fichier excel et valeurs des fichiers BIN (fait dans 1. et 2.)
4. Moyennes écart types des fenêtres?
5. Entrainement de Random Forest : (split en train et test, application de RF au train puis au test)
6. Analyser des résultats (matrices de confusion, accuracy score)

"""
import os
import struct
import csv
import pandas as pd
import numpy as np
import openpyxl
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from agcounts_filter import convert_AC
from datetime import datetime
from scipy.stats import mode

temps_depart = datetime.now()

folder_path = "C:/Users/roman/Documents/BEaCHILD/X_et_Y" 
folder = os.listdir(folder_path)


# Création du fichier csv et enregistrement de l'entête
"""header_row = [['Accelerometer X','Accelerometer Y','Accelerometer Z','Gyroscope X','Gyroscope Y','Gyroscope Z']]
with open("C:/Users/BEaCHILD3/Documents/Stage_Romane/Classification/File_dom.csv", mode="w", newline="", encoding="utf-8") as fichier_csv:
    writer = csv.writer(fichier_csv)
    writer.writerows(header_row)"""

# Initialisation des listes contenant les données des capteurs
X_non_dom = [] 
X_dom = []
idx_X_non_dom = 0
idx_X_dom = 0
idx_Y_dom = 0
idx_Y_non_dom = 0

# Initialisation des listes contenant les annotations vidéos
Y_non_dom = [] 
Y_dom = [] 


#Parcourir les fichiers CSV
for file in folder: #[0:7]
    print(f"On est dans le fichier : {file}")

    # Extension du fichier
    extension = os.path.splitext(file)[1] 

    # Pour les cas ou il y a des annotations avant le start
    allow_start_dom = None
    allow_start_non_dom = None

    """********************************************************************** 1. Lecture des fichiers csv ******************************************************************************"""
    if extension == ".csv":
        
        #/////////////////////////////////////////////////// attention fichier non_dom = [] quand il faut le comparer au données Y (label)

        # Visualisation des données des capteurs
        data_file = pd.read_csv(folder_path + "/" + file, header = 5, names = ["Timestamp","Gyro X","Gyro Y","Gyro Z","Accelerometer X","Accelerometer Y","Accelerometer Z","Event","Quat W","Quat X","Quat Y","Quat Z","None"])    

        # Conversion en AC 
        dom_AC = convert_AC(folder_path + "/" + file)

        # Enregistrement des données des capteurs dans les listes qui seront donnnées au classificateur
        if file[-11:-4] == "non_dom":
            list_dom_AC = dom_AC["AC"].tolist() # Conversion du type pd.serie en type list
            for ac in list_dom_AC :
                X_non_dom.append(ac)

        elif file[-7:-4] == "dom":
            list_dom_AC = dom_AC["AC"].tolist()
            for ac in list_dom_AC:
                X_dom.append(ac)

    """********************************************************** 2. Enregistrer les annotations ***************************************************************************"""

    if extension == ".xlsx":

        start_dom = True
        start_non_dom = True 

        #file_path = "C:/Users/BEaCHILD3/Documents/Stage_Romane/ML/Données validation RCT2/Données validation RCT2/Protocole standardisé_Annotations_BIN_RCT2/Brest/02.10.02/02.10.02_MS_03.xlsx"
        my_wb = openpyxl.load_workbook(folder_path + "/" + file) 
        my_sheet = my_wb.active

        # Initialisation des décalages : sert pour éviter le décalage du aux arrondis des annotations 
        decalage_dom = 0
        decalage_non_dom = 0

        for label in my_sheet["H"]: 

            """*********************************************** Récupération du début de Y ****************************************************"""
            # dom
            # Synchronisation des données X des capteurs et des annotations Y
            if label.value == "Start_RW" :
                if int(round(float(my_sheet.cell(label.row, 12).value))) < float(my_sheet.cell(label.row, 12).value):
                    start_sensor = int(round(float(my_sheet.cell(label.row, 12).value))) + 1
                else:
                    start_sensor = int(round(float(my_sheet.cell(label.row, 12).value))) 
                decalage_dom += start_sensor - float(my_sheet.cell(label.row, 12).value)
                #Ajouter un nb start_sensor de none au début
                for decalage in range(start_sensor):
                    Y_dom.append(None) 
                previous_row_dom = label.row
                # Pour les cas ou il y a des annotations avant le start 
                allow_start_dom = True

            # non_dom 
            # Synchronisation des données X des capteurs et des annotations Y
            if label.value == "Start_LW" :
                if int(round(float(my_sheet.cell(label.row, 12).value))) < float(my_sheet.cell(label.row, 12).value):
                    start_sensor = int(round(float(my_sheet.cell(label.row, 12).value))) + 1
                else:
                    start_sensor = int(round(float(my_sheet.cell(label.row, 12).value))) 
                decalage_non_dom += start_sensor - float(my_sheet.cell(label.row, 12).value)
                #Ajouter un nb start_sensor de none au début
                for decalage in range(start_sensor):
                    Y_non_dom.append(None) 
                previous_row_non_dom = label.row
                # Pour les cas ou il y a des annotations avant le start 
                allow_start_non_dom = True

            """****************************************************** Récupération du contenu de Y ****************************************"""

            if label.value[0:2] == "RW" and allow_start_dom:

                # Récupération de l'index de la ligne
                row = label.row

                # Synchronisation si le start de l'annotation ne correspond pas au stop de l'annotation précédentes
                if my_sheet.cell(row, 12).value != my_sheet.cell(previous_row_dom, 13).value:
                    difference = round(float(my_sheet.cell(row, 12).value)) - round(float(my_sheet.cell(previous_row_dom , 13).value)) 
                    for diff in range(0,difference):
                        Y_dom.append(None) 
                    
                # Synchronisation des données des capteurs avec les annotations
                nb_repetitions = int(round(float(my_sheet.cell(label.row, 13).value))) - int(round(float(my_sheet.cell(label.row, 12).value))) 
                for nb_sec in range(nb_repetitions): 
                    if X_dom and (len(Y_dom) == len(X_dom)) :
                        break
                    if label.value[3:13] == "sédentaire":
                        Y_dom.append("non mouvement")
                    elif label.value[3:7] =="mouv":
                        Y_dom.append("mouvement")
                    elif label.value[3:12] == "non noté":
                        Y_dom.append(None) 

                #Au prochain tour, l'indice de la ligne actuelle sera l'index de la ligne précédente
                previous_row_dom = label.row


            elif label.value[0:2] == "LW" and allow_start_non_dom:
                
                # Récupération de l'index de la ligne
                row = label.row

                # Synchronisation si le start de l'annotation ne correspond pas au stop de l'annotation précédentes
                if my_sheet.cell(row, 12).value != my_sheet.cell(previous_row_non_dom, 13).value:
                    difference = round(float(my_sheet.cell(row, 12).value)) - round(float(my_sheet.cell(previous_row_non_dom , 13).value)) 
                    for diff in range(0,difference):
                        Y_non_dom.append(None) 

                # Synchronisation des données des capteurs avec les annotations
                nb_repetitions = int(round(float(my_sheet.cell(label.row, 13).value))) - int(round(float(my_sheet.cell(label.row, 12).value)))
                for nb_sec in range(nb_repetitions): 
                    if X_non_dom and (len(Y_non_dom) == len(X_non_dom)): 
                        break
                    if label.value[3:13] == "sédentaire":
                        Y_non_dom.append("non mouvement")
                    elif label.value[3:7] =="mouv":
                        Y_non_dom.append("mouvement")
                    elif label.value[3:12] == "non noté":
                        Y_non_dom.append(None)

                #Au prochain tour, l'indice de la ligne actuelle sera l'index de la ligne précédente
                previous_row_non_dom = label.row
        
        # CROPER X A L ALONGUEUR DE Y 
        X_dom = X_dom[:len(Y_dom)]
        X_non_dom = X_non_dom[:len(Y_non_dom)]
        Y_dom = Y_dom[:len(X_dom)]
        Y_non_dom = Y_non_dom[:len(X_non_dom)]
        
        # Au prochain tour, les annotations seront celles de l'enregistrement +1
        idx_Y_dom += 1
        idx_Y_non_dom += 1

        # Couper les données et les annotations à la même taille
        # Code brouillon test : à changer
        """for data in X_dom[idx_X_dom -1]:
            if data.index() > len(Y_dom):
                X_dom[idx_X_dom].enelever le surplus"""

        print(f"len X_dom {len(X_dom)}")
        print(f"len Y_dom {len(Y_dom)}")

temps_fin = datetime.now()

temps = temps_fin - temps_depart
print(f"temps que ca prend : {temps}")

"""****************************************** 5. Entrainement du modèle : (Ex RF : split en train et test, application de RF au train puis au test)***********************************"""
# FAIRE UN TRAIN ET TESTS QUE SUR DROIT ET UN 2e QUE SUR GAUCHE (POHL, 2022) OU FAIRE UN ENSEMBLE GAUCHE DROIT?

# Convertir en array pour faciliter le filtrage
X_array = np.array(X_dom)
Y_array = np.array(Y_dom)
# Créer un masque pour garder uniquement les étiquettes valides (différent de None)
mask = Y_array != None
# Appliquer le masque
X_clean = X_array[mask]
Y_clean = Y_array[mask]


# Aplatir les données (mettre au bon format pour les modèles)
X_flat = np.vstack(X_clean) 
Y_flat = np.hstack(Y_clean) 

# Split data into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X_flat, Y_flat, test_size=0.3, random_state=42)

# Initialize the model
non_dom_Random_Forest = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
non_dom_Random_Forest.fit(X_train, y_train)
# Test the model
y_pred = non_dom_Random_Forest.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy pour RF : {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

print(f"Matrice de confusion RF : {cm}")

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ["mouvement", "non mouvement"])
disp.plot(cmap="Blues")

disp.ax_.set_title("Matrice de Confusion RF")
disp.ax_.set_xlabel("Prédictions")
disp.ax_.set_ylabel("Vérités")
disp.figure_.show()
#********************************* Seuil AC > 75 **************************************
#On a X_dom et Y_dom : construire un Y_pred_2 et le comparer à Y_dom
Y_pred_75 = []

for idx_data in range(0, len(X_clean)-2):
    seuil = X_clean[idx_data] + X_clean[idx_data+1]
    if seuil >= 75 :
        Y_pred_75.append("mouvement")
    elif seuil < 75 :
        Y_pred_75.append("non mouvement")
    else:
        Y_pred_75.append(None)

#Le dernier de Y_pred_75 ne peut pas se joindre à 2 sec avec la valeur d'après
#Y_pred_75.append("non mouvement")

# Evaluate the model
accuracy = accuracy_score(Y_clean, Y_pred_75)
print(f"Accuracy pour seuil > 75 : {accuracy:.2f}")
cm = confusion_matrix(Y_clean, Y_pred_75)
print(f"Matrice de confusion seuil à 75 : {cm}")

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ["mouvement", "non mouvement"])
disp.plot(cmap="Blues")

disp.ax_.set_title("Matrice de Confusion Seuil Ac > 75")
disp.ax_.set_xlabel("Prédictions")
disp.ax_.set_ylabel("Vérités")
disp.figure_.show()

#********************************* Clustering ***************************************
# Supposons que X_clean est ton jeu de données (n_samples, n_features)
k = 2  # nombre de clusters (par exemple : allumé vs éteint)
kmeans = KMeans(n_clusters=k, n_init = "auto", random_state=42)
clusters = kmeans.fit_predict(X_flat)

# Créer un tableau pour mapper chaque cluster à la classe dominante
labels_map = {}
for cluster_id in np.unique(clusters):
    mask = clusters == cluster_id
    majority_class = pd.Series(Y_flat[mask]).mode()[0]
    labels_map[cluster_id] = majority_class
# Traduire les clusters en classes
predicted_labels = np.array([labels_map[c] for c in clusters])

# Evaluate the model
accuracy = accuracy_score(Y_flat, predicted_labels)
print(f"Accuracy pour clustering : {accuracy:.2f}")
cm = confusion_matrix(Y_flat, predicted_labels) # //////////////:: a modifier car cluster 0 peut etre mvt ou noon mvt, il faut leur donner des noms
print(f"Matrice de confusion clustering : {cm}")

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ["mouvement", "non mouvement"])
disp.plot(cmap="Blues")
disp.ax_.set_title("Matrice de Confusion Clustering")
disp.ax_.set_xlabel("Prédictions")
disp.ax_.set_ylabel("Vérités")
disp.figure_.show()
plt.show()