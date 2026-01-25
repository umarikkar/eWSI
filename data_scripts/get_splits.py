import pandas as pd
import os

def get_split_cancer_subtype(data_root, dataset_name, fold_num, istrain):

    if dataset_name == 'tcga-nsclc':
        data_path = [  
            os.path.join(data_root, 'tcga-lusc'),
                            os.path.join(data_root, 'tcga-luad')
                                ]
    else:
        data_path = [os.path.join(data_root, '%s'%dataset_name)]

    if istrain=='train':                                   
        df1 = pd.read_csv('data_scripts/csvs/10foldcv_subtype/%s/train_'%dataset_name + str(fold_num) + '.csv')
        df2 = pd.read_csv('data_scripts/csvs/10foldcv_subtype/%s/val_'%dataset_name + str(fold_num) + '.csv')
        df = pd.concat([df1, df2], ignore_index=True)
    elif istrain=='val': 
        df = pd.read_csv('data_scripts/csvs/10foldcv_subtype/%s/val_'%dataset_name + str(fold_num) + '.csv')
    elif istrain=='test': 
        df = pd.read_csv('data_scripts/csvs/10foldcv_subtype/%s/test_'%dataset_name + str(fold_num) + '.csv')
    else:
        print('invalid name!!!')

    def process_filename(filename):
        filename = filename.split('/')[-1].replace('.pickle', '')
        return filename
    
    df['filename'] = df['filename'].apply(process_filename)

    df = df.rename(columns={'filename':'wsi'})
    print(f"label 0: {sum(df['label']==0)}, label 1: {sum(df['label']==1)}")

    return df, data_path


def get_split_cancer_subtype_kfold(data_root, dataset_name, fold_num, istrain):

    if dataset_name == 'tcga-nsclc':
        data_path = [  
            os.path.join(data_root, 'tcga-lusc'),
                            os.path.join(data_root, 'tcga-luad')
                                ]
    else:
        data_path = [os.path.join(data_root, '%s'%dataset_name)]

    if istrain=='train':                                   
        df = pd.read_csv('data_scripts/csvs/10foldcv_subtype/%s/train_'%dataset_name + str(fold_num) + '.csv')
    elif istrain=='val': 
        df = pd.read_csv('data_scripts/csvs/10foldcv_subtype/%s/val_'%dataset_name + str(fold_num) + '.csv')
    elif istrain=='test': 
        df = pd.read_csv('data_scripts/csvs/10foldcv_subtype/%s/test_'%dataset_name + str(fold_num) + '.csv')
    else:
        print('invalid name!!!')

    def process_filename(filename):
        filename = filename.split('/')[-1].replace('.pickle', '')
        return filename
    
    df['filename'] = df['filename'].apply(process_filename)

    df = df.rename(columns={'filename':'wsi'})
    print(f"label 0: {sum(df['label']==0)}, label 1: {sum(df['label']==1)}")

    return df, data_path


def get_split_molecular_subtype(data_root, dataset_name, fold_num, istrain):

    data_path = [os.path.join(data_root, 'tcga-brca')]
    task = dataset_name.split('-')[-1]

    split_df = pd.read_csv('data_scripts/csvs/other_splits/%s/full/split_%d.csv'%(task, fold_num))
    label_df = pd.read_csv('data_scripts/csvs/tables/%s.csv'%task)

    if istrain=='train':
        # ss = split_df.drop()/
        df = pd.merge(label_df, split_df, left_on='ID', right_on='train')
    elif istrain=='test': 
        df = pd.merge(label_df, split_df, left_on='ID', right_on='test')
    else:
        print('invalid name!!!')

    df = df[label_df.columns]
    df = df.rename(columns={'ID':'wsi'})

    df = df.rename(columns={'label':'label_name'})
    df['label'] = df['label_name'].factorize(sort=True)[0]

    print(f"split: {istrain}, fold: {fold_num}, label 0: {sum(df['label']==0)}, label 1: {sum(df['label']==1)}")

    return df, data_path


def get_split_molecular_subtype_kfold(data_root, dataset_name, fold_num, istrain):

    data_path = [os.path.join(data_root, 'tcga-brca')]
    task = dataset_name.split('-')[-1]

    split_df = pd.read_csv('data_scripts/csvs/other_splits/%s/full/split_train_val_test_%d.csv'%(task, fold_num))
    label_df = pd.read_csv('data_scripts/csvs/tables/%s.csv'%task)

    if istrain=='train':
        # ss = split_df.drop()/
        df = pd.merge(label_df, split_df, left_on='ID', right_on='train')
    elif istrain=='val': 
        df = pd.merge(label_df, split_df, left_on='ID', right_on='val')
    elif istrain=='test': 
        df = pd.merge(label_df, split_df, left_on='ID', right_on='test')
    else:
        print('invalid name!!!')

    df = df[label_df.columns]
    df = df.rename(columns={'ID':'wsi'})

    df = df.rename(columns={'label':'label_name'})
    df['label'] = df['label_name'].factorize(sort=True)[0]

    print(f"split: {istrain}, fold: {fold_num}, label 0: {sum(df['label']==0)}, label 1: {sum(df['label']==1)}")

    return df, data_path


def get_splits_subtype_kfold(data_root, dataset_name, fold_num, istrain):
    if 'mhrd' in dataset_name or 'thrd' in dataset_name or 'tnbc' in dataset_name:
        split_fn = get_split_molecular_subtype_kfold 
    else:
        split_fn = get_split_cancer_subtype_kfold  
    
    df, data_path = split_fn(data_root, dataset_name, fold_num, istrain)

    print(f"split: {istrain}, fold: {fold_num}, label 0: {sum(df['label']==0)}, label 1: {sum(df['label']==1)}")

    return df, data_path

def get_splits_subtype(data_root, dataset_name, fold_num, istrain):
    if 'mhrd' in dataset_name or 'thrd' in dataset_name or 'tnbc' in dataset_name:
        split_fn = get_split_molecular_subtype 
    else:
        split_fn = get_split_cancer_subtype  
    
    df, data_path = split_fn(data_root, dataset_name, fold_num, istrain)

    print(f"split: {istrain}, fold: {fold_num}, label 0: {sum(df['label']==0)}, label 1: {sum(df['label']==1)}")

    return df, data_path