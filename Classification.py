"""
1. Lire les fichiers BIN, conversion dans la bonne unité et transformer en AC
2. Récuperer les annotations
3. Associer annotations fichier excel et valeurs des fichiers BIN (fait dans 1. et 2.)
4. Moyennes écart types des fenêtres?
5. Entrainement de Random Forest : (split en train et test, application de RF au train puis au test)
6. Analyser des résultats (matrices de confusion, accuracy score)


CROP BIEN AU BON ENDROIT
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


folder_path = "C:/Users/BEaCHILD3/Documents/Stage_Romane/ML/donnees_rangees/X_et_Y"
folder = os.listdir(folder_path)


#/////////////////////////////////////////// A voir si on met tout dans le meme fichier ou si on fait 1 fichier pour 1 vidéo : là c'est 1 fichier 1 vidéo ///////////////////////////////////////////
# Création du fichier csv et enregistrement de l'entête
"""header_row = [['Accelerometer X','Accelerometer Y','Accelerometer Z','Gyroscope X','Gyroscope Y','Gyroscope Z']]
with open("C:/Users/BEaCHILD3/Documents/Stage_Romane/Classification/File_dom.csv", mode="w", newline="", encoding="utf-8") as fichier_csv:
    writer = csv.writer(fichier_csv)
    writer.writerows(header_row)"""

# Initialisation des listes contenant les données des capteurs
X_left = []
X_right = []
idx_X_left = 0
idx_X_right = 0
idx_Y_right = 0
idx_Y_left = 0

# Initialisation des listes contenant les annotations vidéos
Y_left = [[]] #//////////////////////////////////////////////////////////// A voir si on a vraiment besoin de rajouter un None
Y_right = [[]]

#Parcourir les fichiers CSV
for file in folder[0:3]:
    print(f"On est dans le fichier : {file}")

    #////////////////////////////////////////////////////////// Selectionne pas les fichiers dans l'ordre croissant (1er fichier = Data_10)
    #print(file[-6:-4])

    # Extension du fichier
    extension = os.path.splitext(file)[1] 

    """********************************************************************** 1. Lecture des fichiers csv ******************************************************************************"""
    if extension == ".csv":
        
        #/////////////////////////////////////////////////// attention fichier left = [] quand il faut le comparer au données Y (label)

        # Visualisation des données des capteurs
        data_file = pd.read_csv(folder_path + "/" + file, header = 5, names = ["Timestamp","Gyro X","Gyro Y","Gyro Z","Accelerometer X","Accelerometer Y","Accelerometer Z","Event","Quat W","Quat X","Quat Y","Quat Z","None"])    

        # Conversion en AC 
        dom_AC = convert_AC(folder_path + "/" + file)


        # Enregistrement des données des capteurs dans les listes qui seront donnnées au classificateur
        if file[-6:-4] == "LW":
            list_dom_AC = dom_AC["AC"].tolist() # Conversion du type pd.serie en type list
            X_left.append(list_dom_AC)
            idx_X_left += 1 

        elif file[-6:-4] == "RW":
            list_dom_AC = dom_AC["AC"].tolist()
            X_right.append(list_dom_AC)
            idx_X_right += 1 
            print(f"len x_right que l'on pop{len(X_right[idx_X_right-1])}")

    """********************************************************************** 2. Enregistrer les annotations ***************************************************************************"""
        #///////////////////// ATTENTION ANNOTATIONS PLUS LONGUES QUE FICHIER (IL FAUT COUPER LES ANNONATIONS A LA LONGUEUR DES DONNEES CSV)

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

        # Iniialisation des décalage : sert pour éviter le décalage du aux arrondis des annotations 
        decalage_right = 0
        decalage_left = 0

        for label in my_sheet["H"]: 

            # Right
            # Synchronisation des données X des capteurs et des annotations Y
            if label.value == "Start_RW" :
                start_sensor = int(round(float(my_sheet.cell(label.row, 12).value)))
                decalage_right += start_sensor - float(my_sheet.cell(label.row, 12).value)
                print(f"Droite : le départ des capteur est en décalage de (nomrlament 7) : {start_sensor}")
                #Ajouter un nb start_sensor de none au début
                for decalage in range(start_sensor):
                    Y_right[idx_Y_right].append(None)
                previous_row_right = label.row

            # Left 
            # Synchronisation des données X des capteurs et des annotations Y
            if label.value == "Start_LW" :
                start_sensor = int(round(float(my_sheet.cell(label.row, 12).value)))
                decalage_left += start_sensor - float(my_sheet.cell(label.row, 12).value)
                #Ajouter un nb start_sensor de none au début
                for decalage in range(start_sensor):
                    Y_left[idx_Y_left].append(None)
                previous_row_left = label.row

                                
            # ////////////////// AJOUT DE LA MAJ DE DECALAGE ET U IF POUR REMETTRE DECALAGE A 0
            #////////////////////////////////////////////// Ajouter une liste dans la liste Y_right et Y_left et pas juste les données directes dans Y_right et Y_left(comme pour les X)
            if label.value[0:2] == "RW":

                # Récupération de l'index de la ligne
                row = label.row

                # Synchronisation de l'allumage des capteurs
                if my_sheet.cell(label.row, 12).value != my_sheet.cell(previous_row_right, 13).value:
                    difference = float(my_sheet.cell(label.row, 12).value) - float(my_sheet.cell(previous_row_right , 13).value)
                    difference_arrondie = int(round(difference))
                    decalage_right += difference_arrondie - difference
                    for diff in range(difference_arrondie):
                        Y_right[idx_Y_right].append(None)

                # Synchronisation des données des capteurs avec les annotations
                else:
                    for nb_sec in range(int(round(float(my_sheet.cell(label.row, 14).value)))):
                        if X_right and (len(Y_right[idx_Y_right]) == len(X_right[idx_X_right-1])) : 
                            break
                        if label.value[3:13] == "sédentaire":
                            Y_right[idx_Y_right].append("non mouvement")
                        elif label.value[3:7] =="mouv":
                            Y_right[idx_Y_right].append("mouvement")
                        elif label.value[3:12] == "non noté":
                            Y_right[idx_Y_right].append(None)

                # Si le décalage est plus important que 1sec, il y a un vrai décalage entre les annotations et les données 
                if decalage_right >= 0.5 :
                    Y_right[idx_Y_right].append("pas meme start et stop")
                    decalage_right = 0
                
                #Au prochain tour, l'indice de la ligne actuelle sera l'index de la ligne précédente
                previous_row_right = label.row
                    
                    

            elif label.value[0:2] == "LW":
                
                # Récupération de l'index de la ligne
                row = label.row

                # Synchronisation de l'allumage des capteurs
                if my_sheet.cell(label.row, 12).value != my_sheet.cell(previous_row_left, 13).value:
                    difference = float(my_sheet.cell(label.row, 12).value) - float(my_sheet.cell(previous_row_left , 13).value)
                    difference_arrondie = int(round(difference))
                    decalage_left += difference_arrondie - difference
                    for diff in range(difference_arrondie):
                        Y_left[idx_Y_left].append(None)

                # Synchronisation des données des capteurs avec les annotations
                else:
                    for nb_sec in range(int(round(float(my_sheet.cell(label.row, 14).value)))):
                        if X_left and (len(Y_left[idx_Y_left]) == len(X_left[idx_X_left-1])) : 
                            break
                        if label.value[3:13] == "sédentaire":
                            Y_left[idx_Y_left].append("non mouvement")
                        elif label.value[3:7] =="mouv":
                            Y_left[idx_Y_left].append("mouvement")
                        elif label.value[3:12] == "non noté":
                            Y_left[idx_Y_left].append(None)

                # Si le décalage est plus important que 1sec, il y a un vrai décalage entre les annotations et les données 
                if decalage_left >= 0.5 :
                    Y_left[idx_Y_left].append(None)
                    decalage_left = 0
                
                #Au prochain tour, l'indice de la ligne actuelle sera l'index de la ligne précédente
                previous_row_left = label.row
        
        # Supprimer les décalage au début entre X et Y
        for annotation in Y_right[idx_Y_right]:
            if annotation == "a_suppr":
                print("l'annotaion est a supprimer")
                Y_right[idx_Y_right].remove(annotation)
        for annotation in Y_left[idx_Y_left]:
            if annotation == "a_suppr":
                Y_left[idx_Y_left].remove(annotation)

        # Au prochain tour, les annotations seront celles de l'enregistrement +1
        idx_Y_right += 1
        idx_Y_left += 1

        # Couper les données et les annotations à la même taille
        # Code brouillon test : à changer
        """for data in X_right[idx_X_right -1]:
            if data.index() > len(Y_right):
                X_right[idx_X_right].enelever le surplus"""

        
        #/////////////////////////////////////////////////////// ATTENTION : X ET Y N'ONT PAS LES MEMES TAILLES!!!!!

        print(f"len Y_left {len(Y_left[idx_Y_left-1])}")

        print(f"len X_right {len(X_right[idx_X_right-1])}")
        #Y_right que pour right
        print(f"len Y_right {len(Y_right[idx_Y_right-1])}")

        deb_val_Y= 125
        deb_val_X= deb_val_Y - 7
        print(f"100 1e données de X_right : {X_right[idx_X_right-1][125:225]}")
        print(f"100 1e annotations de Y_right : {Y_right[idx_Y_right-1][125:225]}")
    

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