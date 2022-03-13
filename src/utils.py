import numpy as np
import random


# Split the tripleList into batches 
def getBatchList(tripleList, num_batches):
    batchSize = len(tripleList) // num_batches
    batchList = [0] * num_batches
    for i in range(num_batches - 1):
        batchList[i] = tripleList[i * batchSize : (i + 1) * batchSize]
    batchList[num_batches - 1] = tripleList[(num_batches - 1) * batchSize : ]
    return batchList

# randomly generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch(tripleList, entity_size):
    newTripleList = [corrupt_head(triple, entity_size) if random.random() < 0.5 
                     else corrupt_tail(triple, entity_size) for triple in tripleList]
    ph, pt ,pr = list(zip(*tripleList))
    nh, nt, nr = list(zip(*newTripleList))
    return ph, pt, pr, nh, nt, nr

def corrupt_head(triple, entity_size):
    return (np.random.randint(entity_size),triple[1],triple[2])

def corrupt_tail(triple, entity_size):
    return (triple[0],triple[1],np.random.randint(entity_size))




def extract_entities(text):
    doc = nlp(text)
    dep_df = pd.DataFrame(columns=['text', 'dep'])
    dep_df['text'] = [tok.text for tok in doc]
    dep_df['dep'] = [tok.dep_ for tok in doc]

    subject_df = dep_df[dep_df['dep'].str.contains('subj')]

    if len(subject_df) == 0:
        # a temporary fix. If there is no subject, skip this sample. 
        return [None, None]
    
    object_df = dep_df[dep_df['dep'].str.contains('obj')]
    modifiers_df = dep_df[dep_df['dep'].str.contains('mod')]


    subject_modifiers = modifiers_df[modifiers_df.index < subject_df.index[0]]
    if subject_modifiers.shape[0] > 0:
        entity1 = " ".join(subject_modifiers['text'].values) + " " + " ".join(subject_df['text'].values)
    else:
        entity1 = " ".join(subject_df['text'].values)

    root_index = dep_df[dep_df['dep'].str.contains('ROOT')].index[0]

    object_modifiers = modifiers_df.iloc[(modifiers_df.index > root_index) &
                            (modifiers_df.index < object_df.index[0])]

    if object_modifiers.shape[0] > 0:

        entity2 = " ".join(object_modifiers['text'].values) + " " + " ".join(object_df['text'].values)
    else:
        entity2 = " ".join(object_df['text'].values)
        
    return [entity1.strip(), entity2.strip()]
