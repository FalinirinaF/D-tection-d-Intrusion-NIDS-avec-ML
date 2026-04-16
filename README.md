# D-tection-d-Intrusion-NIDS-avec-ML
🛡️ NIDS - Network Intrusion Detection System

Ce projet implémente un système de détection d'intrusion réseau utilisant le Machine Learning (Random Forest & XGBoost) sur le dataset NSL-KDD, avec une interface interactive Streamlit.
🛠️ Installation et Configuration

Suivez ces étapes pour installer le projet sur votre machine (testé sur Debian 12 / Kali Linux).
1. Prérequis

Assurez-vous d'avoir Python 3.10+ et pip installés :
Bash

sudo apt update
sudo apt install python3 python3-pip python3-venv

2. Clonage du projet
Bash

git clone https://github.com/FalinirinaF/D-tection-d-Intrusion-NIDS-avec-ML.git
cd "D-tection-d-Intrusion-NIDS-avec-ML"

3. Création de l'environnement virtuel

Il est fortement recommandé d'utiliser un environnement virtuel pour isoler les dépendances :
Bash

# Création
python3 -m venv venv

# Activation
source venv/bin/activate

4. Installation des dépendances

Installez toutes les bibliothèques nécessaires (Pandas, Scikit-learn, XGBoost, Streamlit, etc.) :
Bash

pip install --upgrade pip
pip install -r requirements.txt

5. Lancement de l'application

Pour démarrer l'interface de contrôle et de détection :
Bash

streamlit run app.py

📁 Structure du projet

    app.py : Interface utilisateur Streamlit.

    utils.py : Fonctions de prétraitement et logique de détection.

    models/ : Modèles entraînés (fichiers .joblib ou .pkl).

    data/ : Échantillons du dataset NSL-KDD pour les tests.

📝 Note pour les contributeurs

Le dossier venv/ est exclu du dépôt via .gitignore. Si vous ajoutez de nouvelles bibliothèques, n'oubliez pas de mettre à jour le fichier des dépendances :
Bash

pip freeze > requirements.txt
