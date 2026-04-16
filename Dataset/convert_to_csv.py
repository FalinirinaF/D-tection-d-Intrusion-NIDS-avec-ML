import pandas as pd

# 1. Définir les noms des 43 colonnes du dataset NSL-KDD
# (C'est essentiel pour que votre modèle ML sache quoi analyser)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

def convert_nsl_to_csv(input_file, output_name):
    print(f"Conversion de {input_file} en cours...")
    # Lire le fichier texte (séparateur par défaut est la virgule)
    df = pd.read_csv(input_file, header=None, names=columns)
    
    # Sauvegarder physiquement en CSV
    df.to_csv(output_name, index=False)
    print(f"Succès ! Fichier créé : {output_name}")

# 2. Exécuter la conversion pour l'entraînement et le test
# Assurez-vous que les fichiers .txt sont dans le même dossier que ce script
try:
    convert_nsl_to_csv('KDDTrain+.txt', 'nsl_kdd_train.csv')
    convert_nsl_to_csv('KDDTest+.txt', 'nsl_kdd_test.csv')
except FileNotFoundError as e:
    print(f"Erreur : Vérifiez que le fichier {e.filename} est bien extrait de l'archive.")
