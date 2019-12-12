import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

##########################################################
# Preprocessing Functions
##########################################################

# Function to clean text
def clean(pokemonBio):
    for i in range(len(pokemonBio)):
        pokemonBio[i] = re.sub(r'\n', '', pokemonBio[i])
        pokemonBio[i] = re.sub(r'\.', '', pokemonBio[i])
        pokemonBio[i] = re.sub(r',', '', pokemonBio[i])
        pokemonBio[i] = pokemonBio[i].lower()
    return pokemonBio

# Function to return list of Pokemon types
def getTypeList():
    typeList = ['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost',
                'Steel', 'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark', 'Fairy']
    return typeList

# Function to encode Pokemon's Primary Type
def getPrimType(typeA):
    types = np.zeros((len(typeA), 18))
    for element in [typeA]:
        for i in range(802):
            if 'Normal' == element[i]:
                types[i, 0] = 1
            elif 'Fighting' == element[i]:
                types[i, 1] = 1
            elif 'Flying' == element[i]:
                types[i, 2] = 1
            elif 'Poison' == element[i]:
                types[i, 3] = 1
            elif 'Ground' == element[i]:
                types[i, 4] = 1
            elif 'Rock' == element[i]:
                types[i, 5] = 1
            elif 'Bug' == element[i]:
                types[i, 6] = 1
            elif 'Ghost' == element[i]:
                types[i, 7] = 1
            elif 'Steel' == element[i]:
                types[i, 8] = 1
            elif 'Fire' == element[i]:
                types[i, 9] = 1
            elif 'Water' == element[i]:
                types[i, 10] = 1
            elif 'Grass' == element[i]:
                types[i, 11] = 1
            elif 'Electric' == element[i]:
                types[i, 12] = 1
            elif 'Psychic' == element[i]:
                types[i, 13] = 1
            elif 'Ice' == element[i]:
                types[i, 14] = 1
            elif 'Dragon' == element[i]:
                types[i, 15] = 1
            elif 'Dark' == element[i]:
                types[i, 16] = 1
            elif 'Fairy' == element[i]:
                types[i, 17] = 1
    pd.DataFrame(types, columns=getTypeList()).to_csv('D:/UIP/primType.csv')
    return types

# Function to encode Pokemon's Secondary Type
def getSecondType(typeB):
    types = np.zeros((len(typeB), 18))
    for element in [typeB]:
        for i in range(802):
            if 'Normal' == element[i]:
                types[i, 0] = 1
            elif 'Fighting' == element[i]:
                types[i, 1] = 1
            elif 'Flying' == element[i]:
                types[i, 2] = 1
            elif 'Poison' == element[i]:
                types[i, 3] = 1
            elif 'Ground' == element[i]:
                types[i, 4] = 1
            elif 'Rock' == element[i]:
                types[i, 5] = 1
            elif 'Bug' == element[i]:
                types[i, 6] = 1
            elif 'Ghost' == element[i]:
                types[i, 7] = 1
            elif 'Steel' == element[i]:
                types[i, 8] = 1
            elif 'Fire' == element[i]:
                types[i, 9] = 1
            elif 'Water' == element[i]:
                types[i, 10] = 1
            elif 'Grass' == element[i]:
                types[i, 11] = 1
            elif 'Electric' == element[i]:
                types[i, 12] = 1
            elif 'Psychic' == element[i]:
                types[i, 13] = 1
            elif 'Ice' == element[i]:
                types[i, 14] = 1
            elif 'Dragon' == element[i]:
                types[i, 15] = 1
            elif 'Dark' == element[i]:
                types[i, 16] = 1
            elif 'Fairy' == element[i]:
                types[i, 17] = 1
    pd.DataFrame(types, columns=getTypeList()).to_csv('D:/UIP/secondType.csv')
    return types

# Function to encode Pokemon's Types
def getBothTypes(typeA, typeB):
    types = np.zeros((len(typeA), 18))
    for element in [typeA, typeB]:
        for i in range(802):
            if 'Normal' == element[i]:
                types[i, 0] = 1
            elif 'Fighting' == element[i]:
                types[i, 1] = 1
            elif 'Flying' == element[i]:
                types[i, 2] = 1
            elif 'Poison' == element[i]:
                types[i, 3] = 1
            elif 'Ground' == element[i]:
                types[i, 4] = 1
            elif 'Rock' == element[i]:
                types[i, 5] = 1
            elif 'Bug' == element[i]:
                types[i, 6] = 1
            elif 'Ghost' == element[i]:
                types[i, 7] = 1
            elif 'Steel' == element[i]:
                types[i, 8] = 1
            elif 'Fire' == element[i]:
                types[i, 9] = 1
            elif 'Water' == element[i]:
                types[i, 10] = 1
            elif 'Grass' == element[i]:
                types[i, 11] = 1
            elif 'Electric' == element[i]:
                types[i, 12] = 1
            elif 'Psychic' == element[i]:
                types[i, 13] = 1
            elif 'Ice' == element[i]:
                types[i, 14] = 1
            elif 'Dragon' == element[i]:
                types[i, 15] = 1
            elif 'Dark' == element[i]:
                types[i, 16] = 1
            elif 'Fairy' == element[i]:
                types[i, 17] = 1
    pd.DataFrame(types, columns = getTypeList()).to_csv('D:/UIP/sparseTypes.csv')
    return types

# Expressing the boolean vector as a 1-d array
def densify(types):
    denseTypes = []
    for pokemon in types:
        denseTypes.append(pokemon)
    pd.DataFrame(denseTypes, columns = getTypeList()).to_csv('D:/UIP/denseTypes.csv')
    return denseTypes

# Creating tf-idf matrix
def getTFIDF(biology):
    stopword_list = ['and', 'to', 'the', 'of', 'it']
    freq_vectorizer = TfidfVectorizer(binary=False, stop_words=stopword_list)

    # Get TF-IDF features and store them as CSV
    tfidf = freq_vectorizer.fit_transform(biology)
    pd.DataFrame(tfidf.A).to_csv('D:/UIP/tfidf.csv')
    return tfidf

# Using Latent Semantic Analysis
def getLSA(tfidf):
    svd = TruncatedSVD(n_components=750, n_iter=10, random_state=12345)
    lsa = svd.fit_transform(tfidf)
    print("Proportion of variance preserved = {0:.2f}".format(svd.explained_variance_ratio_.sum()*100))

    # Storing the results
    pd.DataFrame(lsa).to_csv('D:/UIP/LSA.csv')
    return lsa

# Compiling the final dataset
def getFinalData(data, lsa, indexNames):
    finalData = np.append(data, lsa, axis=1)

    # Storing the results
    pd.DataFrame(finalData, index=indexNames).to_csv('D:/UIP/finalData.csv')
    return finalData

##########################################################
# Calling the preprocessing functions
##########################################################

# Load initial dataframe
df = pd.read_csv('D:/UIP/scraping/pokemonfinal.csv', index_col='Name')
dfB = pd.read_csv('D:/UIP/scraping/pokemonfinal.csv')
names = dfB['Name']

# Extracting relevant columns
biology = df['bio']
typeA = df['Type1']
typeB = df['Type2']

getPrimType(typeA)
getSecondType(typeB)
ty = getBothTypes(typeA, typeB)
densify(ty)

# Getting tf-idf and LSA features
tf = getTFIDF(biology)
ls = getLSA(tf)

# Dropping irrelevant columns from Xdata
to_drop = ['Type1', 'Type2', 'bio']
data = df.drop(to_drop, axis=1)

getFinalData(data, ls, names)
