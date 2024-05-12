import spacy
from spacy.tokens import Doc

## GPU will significantly accelerate dependency parsing. Comment out if device does not have GPU support.
spacy.require_gpu()

# Load the SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Access the DependencyParser component
dependency_parser = nlp.get_pipe("parser")

'''
This script does traditional feature engineering, extracting certain features from the data.

Token level features:
    - Neg_type (Categorical):        The type of the negation word (can be affix, single word or multi-word phrase) the token is negated by.
    - Token (String):                Token The token given by the dataset. For the model to learn sentence meaning.
    - Lemma (String):                The lemma of the token given by the dataset.
    - PoS_tag (Categorical):         The POS tag of the token. Give structural information to the model.

Contextual-level features:
    - Same_segment (Bool):           If the token and the negation cue belong to the same segment. 
                                     (This is determined by whether there exists certain punctuation or conjunction between the token and the cue.)
    - Common_ancestor (Categorical): The PoS-tag of the token that is the common ancestor of the token and the cue of the syntactic parsing tree.
    - Distance_to_cue (Numeric):     Distance from the word to the negation cue.
    - Dep_relation (Categorical):    One-hot encoded dependency relation of the token. Provide structural information.
    - Dep_dist_to_cue (Numeric):     The dependency distance to the closest cue token.
    - Dep_path_to_cue (Vector):      The dependency path from the word to the closest cue token.  
'''


def extract_sentence_features(sentence, spaces, neg_cue, parser):

    features = []

    # Some of the common conjunction and punctuation. Case insensitive to use.
    conj_punt_set = [',','.','?','"','\'','!','``',':',';','\'\'','-','--','`','(',')','[',']',
                    'for','and','nor','but','or','yet','while','when','whereas','whenever','wherever','whether','if','because','before',
                     'until','unless','since','so','although','after','as','','Accordingly','After','Also','Besides','Consequently',
                     'Conversely','Finally','Furthermore','Hence','However','Indeed','Instead','Likewise','Meanwhile','Moreover','Nevertheless',
                     'Nonetheless','Otherwise','Similarly','Still','Subsequently','Then','Therefore','Thus','except','rather']

    # Get syntac tree
    tree = [t['parsing_tree'] for t in sentence]
    
    # Get dependency parser tree
    tokens = []
    for t in sentence:
        if type(t['token']) == str:
            tokens.append(t['token'])  
        else:
            tokens.append(' ')
    
    # Negation type
    if neg_cue:
        if len(neg_cue) > 1:
            neg_type = "MULTI"
        elif sentence[neg_cue[0]]['token'] != sentence[neg_cue[0]]['negation_word']:
            neg_type = "AFFIX"
        else: 
            neg_type = "NEG"
    else:
        neg_type = ""

    
    doc = Doc(nlp.vocab, words=tokens, spaces=spaces)
    parser(doc)
    cue_tokens = [token for i, token in enumerate(doc) if i in neg_cue]

    for i, token in enumerate(doc):
        feature_dict = {'token': sentence[i]['token'],     # Feature 1: Token 
                        'neg_type': neg_type,              # Feature 2: Negation type
                        'lemma': sentence[i]['lemma'],     # Feature 3: Lemma
                        'pos_tag': sentence[i]['pos_tag'], # Feature 4: POS tag
                        'is_neg': sentence[i]['negation_word'] != '_' and sentence[i]['negation_word'] != '***', # Feature 5: Is part of the negation cue
                        'same_segment': True,
                        'common_ancester': "",
                        }
        
        # Fetaure 6: Token Cue Distance
        feature_dict['cue_distance'] = 0 if neg_cue == [] else min([abs(sentence[i]['token_id'] - neg_idx) for neg_idx in neg_cue])
        
        # Feature 7: Special Token (conjunction or punctuation) in between
        if neg_cue:        
            if i not in neg_cue:
                if i < neg_cue[0]:
                    for j in range(i, neg_cue[0]):
                        if sentence[j]['token'] in conj_punt_set:
                            feature_dict['same_segment'] = False
                            break
                if i > neg_cue[-1]:
                    for j in range(neg_cue[-1], i):
                        if sentence[j]['token'] in conj_punt_set:
                            feature_dict['same_segment'] = False
                            break
        
        # Feature 7: Common Ancester
        if neg_cue: 
            if i not in neg_cue:    
                # Traverse down to the token and store the ancestors
                token_ancestors = [0]
                for j, node in enumerate(tree):
                    if j == 0:
                        continue
                    
                    if j == i:
                        token_ancestors.append(j)
                        break
                    
                    if '(' in node:
                        token_ancestors.append(j)
                    
                    for _ in range(0, node.count(')')):
                        if len(token_ancestors) > 1:
                            token_ancestors.pop()
                
                # Traverse down to the cue and store the ancestors
                cue_ancestors = [0]
                for j, node in enumerate(tree):
                    if j == 0:
                        continue
                    
                    if j == neg_cue[0]:
                        cue_ancestors.append(j)
                        break
                    
                    if '(' in node:
                        cue_ancestors.append(j)
                    
                    for _ in range(0, node.count(')')):
                        if len(cue_ancestors) > 1:
                            cue_ancestors.pop()
                
                feature_dict['common_ancester'] = sentence[max(set(token_ancestors).intersection(cue_ancestors))]['pos_tag']
            else:
                feature_dict['common_ancester'] = sentence[i]['pos_tag']
    
        # Feature 8: Dependency relation
        feature_dict['dependency_relation'] = token.dep_

        # Create the dependency distance and path vector
        dependency_distance = 0
        dep_path = []
        
        if cue_tokens: 
            # Get the closest cue token
            cue_token, dependency_distance = min([(cue_token, abs(token.i - cue_token.i)) for cue_token in cue_tokens], key=lambda x: x[1])

            # Find the common ancestor of both tokens
            ancestors_token = [token]
            ancestors_idx_token = [token.i]
            ancestors_cue = [cue_token]

            # Traverse from the token to the root, storing ancestors
            current_token = token
            while current_token.head.i is not current_token.i:
                current_token = current_token.head
                ancestors_token.append(current_token)
                ancestors_idx_token.append(current_token.i)

            # Traverse from end_token to the first common ancester
            current_token = cue_tokens[0]
            while current_token.head.i not in ancestors_idx_token:
                current_token = current_token.head
                ancestors_cue.add(current_token)

            # Store the common ancestor
            common_ancestor = current_token

            # Traverse from the token to the common ancestor (going 'up')
            for current_token in ancestors_token:
                dep_path.append(current_token.dep_)
                if current_token.i == common_ancestor.i:
                    break

            # Traverse from cue to the common ancestor (going 'down')
            for current_token in ancestors_cue:
                dep_path.append(current_token.dep_)

        # Feature 9: Dependency distance to Cue
        feature_dict['dependency_distance'] = dependency_distance

        # Feature 10: Dependency path to Cue
        feature_dict['dependency_path'] = 0 if dep_path == [] else dep_path
        
        # Add to features
        features.append(feature_dict)
        
    
    return features