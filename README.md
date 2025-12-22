# Classification

## Ce repositorie contient les codes de plusieurs méthodes de détection du mouvement, le but du projet étant trouver la méthode optimale en comparant les prédictions obtenues avec chaque seuil, aux annotations vidéos recueillies dans l'essai RCT1.

## Les différentes méthodes testées
- Seuil AC>0
- Seuil AC>0 avec fenetres glissantes non chevauchantes de 2s
- Seuil AC>0 avec fenetres glissantes non chevauchantes de 10s
- Seuil AC>2
- Seuil AC>2 avec fenetres glissantes non chevauchantes de 2s
- Seuil AC>2 avec fenetres glissantes non chevauchantes de 10s
- Seuil AC>75 avec fenetres glissantes non chevauchantes de 2s
- Seuil adaptatif de Coley, basé sur la vitesse angulaire (voir rapport mindmaze)
- Puissance > 1 (avec puissance = accX*gyrX + accY*gyrY + accZ*gyrZ (acc = données d'accélération à chaque seconde, et gyr = données de rotation à chaque seconde)
- Puissance > 1 avec fenetre glissantes non chevauchantes de 2s
- Puissance > 1 avec fenetre glissantes non chevauchantes de 10s
- RandomForest sur les Activity Counts (Imbalanced Random Forest avec validation croisée en LeaveOneOut)
- Clustering sur les Activity Counts (test de Kmeans, DBSCAN et GMM, les meilleurs scores venant de KMEANS)

## A changer lors de l'execution
Faire attention aux chemins des fichiers et à l'index des fichiers lus dans le dossier renseigné. Pour lire 1 enregistrement à la fois il faut lire 3 fichiers : le capteur du membre dominant, le capteur du membre non dominant et le fichier d'annotations. Attention aussi lorsqu'il y a récupération des données gyroscopiques : changer l'index des fichiers lus dans la récupération des données d'accélération (au début) ET dans la récupération des données gyroscopiques (plus bas dans le code).

## Résultats et conclusions
Les méthodes AC>0, AC>0 avec fenetre de 2s, AC>2, AC>2 avec fenetre de 2s montrent des résultats généraux bons à excellents, mais une prédiction de la sédentarité plutôt mauvaise (résultats retrouvés dans la littérature). Les méthodes de Puissance>1, Puissance>1 avec fenetre de 2s, et Random Forest montrent aussi de bons résultats généraux mais des prédictions de sédentarité très mauvais.
En revanche, les méthodes AC>75, le clustering et le seuil adaptatif de Coley présentent des taux d'accuracy moyens, bien inférieurs aux autres méthodes.
Les fenetres glissantes de 10 secondes ne paraissent pas non plus adaptée au public : elles prédisent trop de mouvement, inhibant les secondes de sédentarité qui sont souvent bien plus courtes que 10s. Les scores généraux sont satisfaisants, mais les scores de prédictions de la sédentarité sont bien inférieurs aux autres méthodes.

## Structures des codes
La même méthode est appliquée pour tester les différents seuils :
1. Récupération des données actimétriques, transformation en Activty Counts et synchronisation avec les annotations vidéos. Les annotations étant définies au millième de seconde près, un vote à la majorité est effectué pour réduire à une annotation par seconde. Par exemple, si l'annotation "mouvement" est présente jusque la seconde 10,786 : la seconde 10 est définie comme "mouvement", car le temps dépasse la moitié d'une seconde. L'affiche graphique permet ensuite de vérifier la bonne synchronisation des données.
2. Nettoyage des données et des annotations : suppression des secondes sans information sur le mouvement (annotations "non noté" et secondes représentant le décalage entre 2 annotations ou en extrémité d'enregistrement).
3. Facultatif suivant la méthode : récupération et nettoyage des données gyroscopiques.
4. Création et prédiction du seuil
5. Calcul des métriques : Accuracy et F1 scores (F1 mouvement, F1 non mouvement, F1 weighted, F1 macro)





