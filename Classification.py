"""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from agcounts_filter import convert_AC


folder_path = "C:/Users/BEaCHILD3/Documents/Stage_Romane/ML/donnees_rangees"
folder = os.listdir(folder_path)

#/////////////////////////////////////////// A voir si on met tout dans le meme fichier ou si on fait 1 fichier pour 1 vidéo : là c'est 1 fichier 1 vidéo ///////////////////////////////////////////
# Création du fichier csv et enregistrement de l'entête
header_row = [['Accelerometer X','Accelerometer Y','Accelerometer Z','Gyroscope X','Gyroscope Y','Gyroscope Z']]
with open("C:/Users/BEaCHILD3/Documents/Stage_Romane/Classification/File_dom.csv", mode="w", newline="", encoding="utf-8") as fichier_csv:
    writer = csv.writer(fichier_csv)
    writer.writerows(header_row)


#Parcourir les fichiers BIN
for file in folder[0:3]:
    print(f" on est dans le fichier : {file}")
    Y_left = [None]
    Y_right = [None]
    #////////////////////////////////////////////////////////// Selectionne pas les fichiers dans l'ordre croissant (1er fichier = Data_10)
    #print(file[-6:-4])

    # Extension du fichier
    extension = os.path.splitext(file)[1] 

    """********************************************************************** 1. Lecture des fichiers BIN ******************************************************************************"""
    if extension == ".csv":
        X_left = []
        X_right = []
        #/////////////////////////////////////////////////// attention fichier left = [] quand il faut le comparer au données Y (label)

        # Visualisation des données des capteurs
        data_file = pd.read_csv(folder_path + "/" + file, header = 5, names = ["Timestamp","Gyro X","Gyro Y","Gyro Z","Accelerometer X","Accelerometer Y","Accelerometer Z","Event","Quat W","Quat X","Quat Y","Quat Z","None"])    
        #print(data_file)

        # Conversion en AC 
        dom_AC = convert_AC(folder_path + "/" + file)

        # Enregistrement des données des capteurs dans les listes qui seront donnnées au classificateur
        if file[-6:-4] == "LW":
            X_left.append(dom_AC["AC"])

        elif file[-6:-4] == "RW":
            X_right.append(dom_AC["AC"])
        
    
        #///////////////////////////////////////// Calcul 2x l'AC : pq pas faire une boucle pour calculer right et left en UL

    """********************************************************************** 2. Enregistrer les annotations ***************************************************************************"""
        #///////////////////// ATTENTION ANNOTATIONS PLUS LONGUES QUE FICHIER (IL FAUT COUPER LES ANNONATIONS A LA LONGUEUR DES DONNEES BIN)

        #for file in folder[:2]:
            #print(file[-6:-4])
            # Extension du fichier
            #extension = os.path.splitext(file)[1] """

    
    if extension == ".xlsx":

        start_right = True
        start_left = True 

        #file_path = "C:/Users/BEaCHILD3/Documents/Stage_Romane/ML/Données validation RCT2/Données validation RCT2/Protocole standardisé_Annotations_BIN_RCT2/Brest/02.10.02/02.10.02_MS_03.xlsx"
        my_wb = openpyxl.load_workbook(folder_path + "/" + file) 
        my_sheet = my_wb.active

        # A CORRIGER NE MARCHE PAS (PROBLEME AVEC INSERT SUR LA SERIE PANDAS)
        """for label in my_sheet["H"]: 
             # Synchronisation des données X des capteurs et des annotations Y
            if label.value == "Start_RW":
                for sec_decalage in range(int(round(float(my_sheet.cell(label.row, 12).value)))):
                    # Ajout de N 0 au début de la liste des données X
                    X_right[0].insert(0,0)"""

            elif label.value[0:2] == "RW":
                for nb_sec in range(int(round(float(my_sheet.cell(label.row, 14).value)))):
                    if X_right and (len(Y_right) == len(X_right[0])) :
                        break
                    if label.value[3:13] == "sédentaire":
                        Y_right.append("non mouvement")
                    elif label.value[3:7] =="mouv":
                        Y_right.append("mouvement")
                    elif label.value[3:12] == "non noté":
                        Y_right.append(None)

                    
                    if start_right == True and X_right and len(Y_right) > 1 :
                        second_to_start = my_sheet.cell(label.row, 12).value
                        #print(second_to_start)
                        for idx_value in range(int(round(float(second_to_start)))):
                            X_right[0].pop(idx_value)
                        start_right = False
                    
                    


            elif label.value[0:2] == "LW":
                for nb_sec in range(int(round(float(my_sheet.cell(label.row, 14).value)))):
                    if X_left and (len(Y_left) == len(X_left[0])) : 
                        break
                    if label.value[3:13] == "sédentaire":
                        Y_left.append("non mouvement")
                    elif label.value[3:7] =="mouv":
                        Y_left.append("mouvement")
                    elif label.value[3:12] == "non noté":
                        Y_left.append(None)

                    # ///////////////////////////// SUPPRIMER LES N PREMIERES VALEURS DE X CAR LES ANNOTATIONS NE COMMENCENT PAS AU TEMPS 0 ///////////////////////////////////////
                    if start_left == True and X_left and len(Y_left) > 1 :
                        second_to_start = my_sheet.cell(label.row, 12).value
                        for idx_value in range(int(round(float(second_to_start)))):
                            X_left[0].pop(idx_value)
                        start_left = False
        
        #/////////////////////////////////////////////////////// ATTENTION : X ET Y N'ONT PAS LES MEMES TAILLES!!!!!

        #print(f"len X_left {len(X_left[0])}")
        print(f"len Y_left {len(Y_left)}")

        print(f"len X_right {len(X_right[0])}")
        #Y_right que pour right
        print(f"len Y_right {len(Y_right)}")
        #print(f"len_sensor_data {len_sensor_data}")"""
"""***************************************************************** 4. Moyennes écart types des fenêtres?***************************************************************************"""

"""****************************************** 5. Entrainement du modèle : (Ex RF : split en train et test, application de RF au train puis au test)***********************************"""
# FAIRE UN TRAIN ET TESTS QUE SUR DROIT ET UN 2e QUE SUR GAUCHE (POHL, 2022) OU FAIRE UN ENSEMBLE GAUCHE DROIT?
"""
# Split data into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X_right, Y_right, test_size=0.3, random_state=42)

# Initialize the model
non_dom_Random_Forest = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
non_dom_Random_Forest.fit(X_train, y_train)
# Test the model
y_pred = non_dom_Random_Forest.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.ax_.set_title("Matrice de Confusion")
disp.ax_.set_xlabel("Prédictions")
disp.ax_.set_ylabel("Vérités")
disp.figure_.show()
"""