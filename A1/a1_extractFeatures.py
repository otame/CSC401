import numpy as np
import argparse
import json
import re
import pandas as pd
from tqdm import tqdm

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
BGL = BGL[["WORD", "AoA (100-700)", "IMG", "FAM"]]
BGL.dropna(inplace=True, subset=["WORD"])
BGL.fillna(0, inplace=True)
# BGL = BGL.values
War = pd.read_csv('u/cs401/Wordlists/Ratings_Warriner_et_al.csv')
War = War[['Word', "V.Mean.Sum", 'A.Mean.Sum', 'D.Mean.Sum']]
War.dropna(inplace=True, subset=["Word"])
War.fillna(0, inplace=True)

left_f = open("/u/cs401/A1/feats/Left_IDs.txt")
left_liwc = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
right_f = open("/u/cs401/A1/feats/Right_IDs.txt")
right_liwc = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")
center_f = open("/u/cs401/A1/feats/Center_IDs.txt")
center_liwc = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
alt_f = open("/u/cs401/A1/feats/Alt_IDs.txt")
alt_liwc = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")


left_ids = left_f.readlines()
left_ids = [i.strip() for i in left_ids]

right_ids = right_f.readlines()
right_ids = [i.strip() for i in right_ids]

center_ids = center_f.readlines()
center_ids = [i.strip() for i in center_ids]

alt_ids = alt_f.readlines()
alt_ids = [i.strip() for i in alt_ids]

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    feats = np.zeros(29, dtype=float)
    feats[0] = len(re.findall(r"\b[A-Z][A-Z][A-Z]+/", comment))
    comment = comment.lower()
    feats[1] = len(re.findall(r'\b' + r'/|\b'.join(FIRST_PERSON_PRONOUNS) + '/', comment))
    feats[2] = len(re.findall(r'\b' + r'/|\b'.join(SECOND_PERSON_PRONOUNS) + '/', comment))
    feats[3] = len(re.findall(r'\b' + r'/|\b'.join(THIRD_PERSON_PRONOUNS) + '/', comment))
    feats[4] = len(re.findall(r'/cc\b', comment))
    feats[5] = len(re.findall(r'/vbd\b', comment))
    feats[6] = len(re.findall(r"'ll/|\bwill/|\bgonna/|\bgo/vbg to/to [a-z]+/vb", comment))
    feats[7] = len(re.findall(r',/,', comment))
    feats[8] = len(re.findall(r'/nfp\b', comment))
    feats[9] = len(re.findall(r'/nns?\b', comment))
    feats[10] = len(re.findall(r'/nnps?\b', comment))
    feats[11] = len(re.findall(r'/rb|/rbr|/rbs', comment))
    feats[12] = len(re.findall('/wdt|/wp|/wp\$|/wrb', comment))
    feats[13] = len(re.findall(r'\b' + r'/|\b'.join(SLANG) + '/', comment))
    tokens = re.findall(r'\b[a-z]+/', comment)
    tokens = [x[:-1] for x in tokens]
    feats[16] = len(re.findall(r'\n', comment))
    feats[14] = len(tokens)/feats[16]
    if len(tokens) != 0:
        feats[15] = len("".join(tokens))/len(tokens)
        BGL_words = BGL[BGL["WORD"].str.match(r'^' + r"$|^".join(tokens) + r'$')]
        feats[17] = BGL_words["AoA (100-700)"].mean()
        feats[18] = BGL_words["IMG"].mean()
        feats[19] = BGL_words["FAM"].mean()
        feats[20] = BGL_words["AoA (100-700)"].std()
        feats[21] = BGL_words["IMG"].std()
        feats[22] = BGL_words["FAM"].std()
        War_words = War[War["Word"].str.match(r'^' + r"$|^".join(tokens) + r'$')]
        feats[23] = War_words['V.Mean.Sum'].mean()
        feats[24] = War_words['A.Mean.Sum'].mean()
        feats[25] = War_words['D.Mean.Sum'].mean()
        feats[26] = War_words['V.Mean.Sum'].std()
        feats[27] = War_words['A.Mean.Sum'].std()
        feats[28] = War_words['D.Mean.Sum'].std()
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

    if comment_class == "Left":
        feats[29:173] = left_liwc[left_ids.index(comment_id)]
        feats[173] = 0
    elif comment_class == "Center":
        feats[29:173] = center_liwc[center_ids.index(comment_id)]
        feats[173] = 1
    elif comment_class == "Right":
        feats[29:173] = right_liwc[right_ids.index(comment_id)]
        feats[173] = 2
    elif comment_class == "Alt":
        feats[29:173] = alt_liwc[alt_ids.index(comment_id)]
        feats[173] = 3

    return feats


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    i = 0
    for sent in tqdm(data):
        feats[i, :29] = extract1(sent['body'])
        feats[i, :] = extract2(feats[i, :], sent['cat'], sent['id'])
        i += 1
    feats = np.nan_to_num(feats)
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

