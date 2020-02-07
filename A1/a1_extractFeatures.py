import numpy as np
import argparse
import json
import re
import pandas as pd

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

BGL = pd.read_csv('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv')
BGL.dropna(1, inplace=True, how="all")
BGL.dropna(inplace=True, subset=["WORD"])
BGL.fillna(0)
War = pd.read_csv('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv')
War.dropna(1, inplace=True, how="all")
War.dropna(inplace=True, subset=["Word"])
War.fillna(0)

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    feats = np.empty(29, dtype=float)
    feats[0] = len(re.findall(r"\b[A-Z][A-Z][A-Z]+/", comment))
    for i in FIRST_PERSON_PRONOUNS:
        feats[1] += len(re.findall(r'\b' + i + '/', comment))
    for i in SECOND_PERSON_PRONOUNS:
        feats[2] += len(re.findall(r'\b' + i + '/', comment))
    for i in THIRD_PERSON_PRONOUNS:
        feats[3] += len(re.findall(r'\b' + i + '/', comment))
    feats[4] = len(re.findall(r'/CC\b', comment))
    feats[5] = len(re.findall(r'/VBD\b', comment))
    feats[6] = len(re.findall(r"'ll/|\bwill/|\bgonna/|\bgoing/VBG to/TO [a-z]+/VB", comment))
    feats[7] = len(re.findall(r'\b,/,\b', comment))
    feats[8] = len(re.findall(r'/NFP\b', comment))
    feats[9] = len(re.findall(r'/NNS?\b', comment))
    feats[10] = len(re.findall(r'/NNPS?\b', comment))
    feats[11] = len(re.findall(r'/RB|/RBR|/RBS', comment))
    feats[12] = len(re.findall('/WWDT|/WP|/WP\$|/WRB', comment))
    for i in SLANG:
        feats[13] += len(re.findall(r'\b' + i + '/', comment))
    feats[14] = len(comment)
    tokens = re.findall(r'\b[A-Za-z]+/', comment)
    tokens = [x[:-1].lower() for x in tokens]
    feats[16] = len(re.findall(r'\n', comment))
    if len(tokens) != 0:
        feats[15] = len("".join(tokens))/len(tokens)
        BGL_words = BGL[BGL["WORD"].str.match("|".join(tokens))]
        feats[17] = BGL_words.iloc[:, 3].mean()
        feats[18] = BGL_words.iloc[:, 4].mean()
        feats[19] = BGL_words.iloc[:, 5].mean()
        feats[20] = BGL_words.iloc[:, 3].std()
        feats[21] = BGL_words.iloc[:, 4].std()
        feats[22] = BGL_words.iloc[:, 5].std()
        War_words = War[War["Word"].str.match("|".join(tokens))]
        feats[23] = War_words.iloc[:, 2].mean()
        feats[24] = War_words.iloc[:, 5].mean()
        feats[25] = War_words.iloc[:, 8].mean()
        feats[26] = War_words.iloc[:, 2].std()
        feats[27] = War_words.iloc[:, 5].std()
        feats[28] = War_words.iloc[:, 8].std()
    return feats
    
    
def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''

    f = open("/u/cs401/A1/feats/" + comment_class + "_IDs.txt")
    liwc = np.load("/u/cs401/A1/feats/" + comment_class + "_feats.dat.npy")
    ids = f.readlines()
    ids = [i.strip() for i in ids]
    feats[29:172] = liwc[ids.index(comment_id)]
    feats[173] = comment_class

    return feats


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    i = 0
    for sent in data:
        feats[i, :29] = extract1(sent['body'])
        feats[i, :] = extract2(feats[i, :], sent['cat'], sent['id'])
        i += 1

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

