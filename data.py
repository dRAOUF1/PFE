import pandas as pd
import numpy as np
import random
from App import App


app = App("http://localhost:3001/getEmbeddings/all/all")

# Supposons que vous ayez votre liste d'embeddings : embedding_list
embedding_list = app.embeddings  # Remplacez les points de suspension par vos vecteurs

num_embeddings = len(embedding_list)
num_pairs = 100  # Définissez le nombre de paires que vous voulez créer de manière aléatoire


# Créez une liste pour stocker les vecteurs et les classes
data = []

# Parcourez les embeddings de la liste
for _ in range(num_pairs):
    # Choisissez aléatoirement deux indices distincts
    idx1, idx2 = np.random.choice(num_embeddings, size=2, replace=False)
    emb1 = embedding_list[idx1]
    emb2 = embedding_list[idx2]
    
    # Vérifiez si les embeddings sont égaux
    if np.array_equal(emb1, emb2):
        # S'ils sont égaux, ajoutez-les à la liste des données avec la classe 1
        data.append([emb1, emb2, 1])
    else:
        # Sinon, ajoutez-les à la liste des données avec la classe 0
        data.append([emb1, emb2, 0])

# Convertissez la liste des données en DataFrame pandas
df = pd.DataFrame(data, columns=['Vecteur 1', 'Vecteur 2', 'Classe'])

# Affichez les premières lignes du DataFrame
print(df.head())