import pandas as pd
import numpy as np
from pyjarowinkler.distance import get_jaro_winkler_similarity

def clean_data(df: pd.DataFrame,
                select_col: str,
                token_col: str):
    """
    Clean and normalize text data in a DataFrame column.
    """
    df_output = df.copy()

    # lowercase + ensure string + fill nulls
    df_output[token_col] = (
        df_output[select_col]
        .fillna("")              # avoid NaN
        .astype(str)             # ensure string type
        .str.lower()
    )

    # remove non-alphanumeric characters
    df_output[token_col] = df_output[token_col].str.replace(
        r'[^a-zA-Z0-9]',
        ' ',
        regex=True
    )

    # collapse sequences of single-letter tokens: e.g., "w l a" -> "wla"
    df_output[token_col] = df_output[token_col].str.replace(
        r'\b(?:[a-z]\s+){1,}[a-z]\b',
        lambda m: m.group(0).replace(" ", ""),
        regex=True
    )

    # pad numbers with spaces
    df_output[token_col] = df_output[token_col].str.replace(
        r'(\d+\.?\,?\d*)',
        r' \1 ',
        regex=True
    )

    # normalize whitespace by splitting and rejoining
    df_output[token_col] = (
        df_output[token_col]
        .str.split()
        .apply(lambda tokens: " ".join(tokens) if tokens else "")
    )

    return df_output


def exact_match(reference_df, df2, col_df_1_for_comparison, col_df_1_for_labels, col_df_2, col_df2_label):
    # create copies and prepare lowercase comparison columns
    
    df_1_trimmed = reference_df.drop_duplicates(col_df_1_for_comparison)[[col_df_1_for_comparison, col_df_1_for_labels]].copy()
    reference_df_cols = df_1_trimmed.columns.tolist()
    df_2_trimmed = df2[df2['Manufacturer_label'].isnull()].copy()
    df_2_prelabelled = df2[df2['Manufacturer_label'].notnull()].copy()

    matches = df_1_trimmed.merge(
        df_2_trimmed,
        left_on=col_df_1_for_comparison,
        right_on=col_df_2,
        how='right',
    )

    matches[col_df2_label] = np.where(matches[col_df_1_for_labels].notnull(), matches[col_df_1_for_labels], None)
    matches['level'] = np.where(matches[col_df_1_for_labels].notnull(), 'exact_match_' + col_df_1_for_comparison, None)

    df_devices = pd.concat([matches, df_2_prelabelled]).drop(columns = reference_df_cols)

    return df_devices

def best_match_jw(name, df_to_match, col_to_match, col_labels):
    """
    Find the best matching supplier for a given name using Jaro-Winkler similarity.

    Parameters:
        name (str): The name to match against the supplier list.
        supplier_list (list of str): A list of supplier names to compare against.

    Returns:
        tuple: A tuple containing the best matching supplier (str) and the best score (float).
               If no match is found, returns (None, -1).
    """
    df_to_match = df_to_match.copy().drop_duplicates(col_to_match)[[col_to_match, col_labels]]
    dict_to_match = df_to_match.set_index(col_to_match)[col_labels].to_dict()
    list_to_match = list(dict_to_match.keys())

    best_match = None
    best_score = -1
    for supplier in list_to_match:
        score = get_jaro_winkler_similarity(name, supplier)
        if score > best_score:
            best_score = score
            best_match = dict_to_match[supplier]
    return best_match, best_score

def jaro_winkler_match(logger, df_to_match, df_reference, label_col_name, score_col_name, col_to_match, reference_col, reference_labels_col, score_threshold=0.86):
    df_to_jw_match = df_to_match[df_to_match[label_col_name].isnull()]
    df_matched_previously = df_to_match[df_to_match[label_col_name].notnull()]
    # find best supplier matches using Jaro-Winkler similarity
    df_to_jw_match[[label_col_name, score_col_name]] = df_to_jw_match[col_to_match].apply(
        lambda x: pd.Series(best_match_jw(x, df_reference, reference_col, reference_labels_col))
    )

    # filter matches with a score above the threshold
    df_jw_match = df_to_jw_match[df_to_jw_match[score_col_name] >= score_threshold]
    df_rest = df_to_jw_match[df_to_jw_match[score_col_name] < score_threshold]

    logger.info("Jaro-Winkler matches found above threshold: {}", len(df_jw_match))
    logger.info("Remaining to be matched: {}", len(df_rest))
    
    df_output = pd.concat([df_matched_previously, df_jw_match, df_rest])
    
    return df_output

def number_of_tokens_match(name_tokens, supplier_list):
    """
    Calculates the best matching supplier based on the number of common tokens 
    between the provided name tokens and the suppliers' tokens.

    Parameters:
        name_tokens (list): A list of tokens representing the name to match.
        supplier_list (dict): A dictionary where keys are supplier names and 
                              values are lists of tokens associated with each supplier.

    Returns:
        tuple: 
            - If there are multiple best matches:
                - list: A list of suppliers that have the highest matching score.
                - int: The score of the best match (always 0 in this case).
            - If there is a single best match:
                - str: The name of the best matching supplier.
                - float: The score of the best match.
                - bool: Indicates whether there were multiple matches (False).
    """
    best_match = None
    best_score = 0
    multiple_matches = False
    best_match_list = []
    for supplier in supplier_list.keys():
        common_tokens = list(set(name_tokens) & set(supplier_list[supplier]))
        try:
            score = len(common_tokens)/len(supplier_list[supplier])
        except ZeroDivisionError:
            score = 0
        if score > best_score:
            best_score = score
            best_match = supplier
            best_match_list = [supplier]
        elif score == best_score and score != 0:
            multiple_matches = True
            best_match_list.append(supplier)
    if multiple_matches:
        best_score = 0
        return best_match_list, best_score
    return best_match, best_score, multiple_matches
