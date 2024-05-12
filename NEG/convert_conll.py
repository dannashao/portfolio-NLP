import pandas as pd
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(
                    prog='convert_conll',
                    description='This script convert 16 or 19 column conll file to 10 column.',
                    epilog='Example usage: python3 convert_conll.py dev_to_annotate converted.conll 16. DO NOT ADD \'\' OR , WHEN CALLING.')
parser.add_argument('inputfile', help="Input path for the 16 or 19 column conll")  
parser.add_argument('output_name', help="Output filename")
parser.add_argument('col_num',type=int, help="The number of column the data have. Can be 16 or 19.")


def read_conll(filename):
    '''
    Read data with conll format and transform it into a DataFrame
    '''
    column_names = ['document_id', 'sentence_id', 'token_id', 'token', 'lemma', 'pos_tag', 'parsing_tree', 'negation_word', 'negation_scope', 'negation_event']
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if line:
            columns = line.split('\t')
            data.append(columns)
    df = pd.DataFrame(data, columns=column_names)
    return df


def insert_column(x,y,dfc, df, range_array, extended):
    '''
    Support function for converting the conll file.
    This function helps to correctly insert the repetead sentences (sentence with multiple negations)
    '''
    insert_sentence = df.iloc[:, range_array].loc[(df[0]==x) & (df[1]==y)]          # The copied sentence
    insert_sentence.columns = range(insert_sentence.columns.size)                   # Reset column name
    insert_at = insert_sentence.tail(1).index[0]+1                                  # Index where to insert

    dfc = pd.concat([dfc.iloc[:insert_at+extended], insert_sentence, dfc.iloc[insert_at+extended:]],ignore_index=True)
    extended += len(insert_sentence.index)
    return dfc, extended

def convert_conll(df, column_num):
    '''
    Convert the 16 or 19 column rows to 10 column.
    '''
    df2 = df.loc[~df[10].str.contains("_")]
    df3 = df.loc[~df[13].str.contains("_")]

    dfc = df.iloc[:, 0:10]
    extended = 0
    for x, y in set(zip(df2[0], df2[1])):
        dfc, extended = insert_column(x,y,dfc, df, np.r_[0:7, 10,11,12],extended)

    extended = 0
    for x, y in set(zip(df3[0], df3[1])):
        dfc, extended = insert_column(x,y,dfc, df, np.r_[0:7, 13,14,15],extended)

    if column_num == 19:
        df4 = df.loc[~df[16].str.contains("_")]
        extended = 0
        for x, y in set(zip(df4[0], df4[1])):
            dfc, extended = insert_column(x,y,dfc, df, np.r_[0:7, 16,17,18],extended)

    return dfc

def rearrange_faulty_sentence(df):
    sentences=[]
    for x, y in sorted(set(zip(df[0], df[1]))):
        sentence_df = df.loc[(df[0]==x) & (df[1]==y)]
        sentences.append(sentence_df)

    df = pd.concat(sentences)
    return df


def main(inputfile, outputfile, column_num):
    df_names = list(range(column_num))
    df = pd.read_csv(inputfile, sep='\t',header=None,names=df_names) # 16 or 19

    df = df.fillna('_')
    dfc = convert_conll(df,column_num)
    dfc = rearrange_faulty_sentence(dfc)
    dfc.to_csv(outputfile, sep="\t", index=False, header=False)
    
if __name__ == '__main__':
    args = parser.parse_args()
    input_file_path = args.inputfile
    output_file_name = args.output_name
    column_num = args.col_num
    main(input_file_path, output_file_name, column_num)