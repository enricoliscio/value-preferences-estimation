import pandas as pd

value_labels = ['kosten', 'land', 'leiding', 'samen', 'zelf']
dataset_df_columns = ['question_id', 'participant_id', 'dutch', 'english'] + value_labels
motivations_files = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']

accent_table = {
    'Ã¢': 'â',
    'Ãª': 'ê',
    'Ã»': 'û',
    'Ã«': 'ë',
    'Ã¶': 'ö',
    'Ã¯': 'ï',
    'Ã©': 'é',
    'Ã¨': 'è',
    'Ã¡': 'á',
    'Ã¼': 'ü',
    'Ãš': 'ù',
    'Ãº': 'ú',
    'Ã‰': 'é',
    'Ã³': 'ó',
    'dÃ­t': 'dat',
    'â€™': "'",
    'â€˜': "'",
    'â€œ': "'",
    'â€ž': '"',
    'â€¦': '',
    'â€': "'",
    '\n': ' ',
    'â‚¬': '€',
    'Â´': "'",
    '_x000D_': ' '
}


def create_sample_annotation(question_id, row):
    """ Create new dataframe line with index <question_id>_<participant_id>.
    """
    index = [f"{question_id}_{int(row['id'])}"]
    df = pd.DataFrame(columns=dataset_df_columns, index=index)
    df.loc[index] = [question_id, int(row['id']), row['motivation'], row['english']] \
                    + row[value_labels].astype(int).to_list()
    return df


def load_data_from_source():
    """ Load data that is needed for NLP from source Excel files.
    """
    df = pd.DataFrame(columns=dataset_df_columns)

    for question_id, file_name in enumerate(motivations_files, 1):
        xls = pd.ExcelFile(f'./value_profiles/data/{file_name}.xlsx', engine='openpyxl')
        source_df = pd.read_excel(xls, 'Data', engine='openpyxl')

        for _, row in source_df.dropna().iterrows():
            if len(row['motivation']) < 2:
                continue
            new_sample_df = create_sample_annotation(question_id, row)
            df = pd.concat([df, new_sample_df])

    return df


def clean_dutch_data(text):
    for weird_stuff in accent_table.keys():
        if weird_stuff in text:
            text = text.replace(weird_stuff, accent_table[weird_stuff])
    return text


def clean_data():
    """ Only implemented for Dutch.
    """
    data = load_data_from_source()
    data.loc[:, 'dutch'] = data.dutch.apply(clean_dutch_data)

    return data


def clean_and_dump_data(dest='./value_profiles/data/clean_data.xlsx'):
    df = clean_data()
    df.to_excel(dest)


def load_motivations(file_path='./value_profiles/data/translated_clean_data.csv'):
    return pd.read_csv(file_path, index_col=['id'])
