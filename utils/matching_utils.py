import pandas as pd
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


def exact_match(reference_df, df2, col_df_1_for_comparison, col_df_1_for_labels, col_df_2):
    """
    Performs an exact match between two DataFrames based on lowercase comparison of specified columns.
    
    This function creates lowercase versions of the specified columns from both DataFrames and performs
    a right join to find exact matches. The function returns only rows where a match was found in the
    reference DataFrame.
    
    Parameters
    ----------
    reference_df : pd.DataFrame
        The reference DataFrame containing the data to match against.
    df2 : pd.DataFrame
        The second DataFrame to match with the reference DataFrame.
    col_df_1_for_comparison : str
        The column name in reference_df to use for comparison.
    col_df_1_for_labels : str
        The column name in reference_df to use for labels (included in output).
    col_df_2 : str
        The column name in df2 to use for comparison.
    
    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing rows where exact matches (case-insensitive) were found.
        Rows where col_df_1_for_comparison is null are excluded. Lowercase comparison columns
        are removed from the output.
    
    Notes
    -----
    - The comparison is case-insensitive, performed on lowercase versions of the columns.
    - Duplicates in the reference DataFrame are removed before processing.
    - Column suffixes '_ref' and '_other' are applied to overlapping column names.
    """
    # create copies and prepare lowercase comparison columns
    df_1_trimmed = reference_df[[col_df_1_for_comparison, col_df_1_for_labels]].drop_duplicates().copy()
    df_2_trimmed = df2.copy()

    col1_lower = f'{col_df_1_for_comparison}_lower'
    col2_lower = f'{col_df_2}_lower'

    df_1_trimmed[col1_lower] = df_1_trimmed[col_df_1_for_comparison].str.lower()
    df_2_trimmed[col2_lower] = df_2_trimmed[col_df_2].str.lower()

    # do an inner join on the prepared lowercase token columns to get exact token matches
    matches = df_1_trimmed.merge(
        df_2_trimmed,
        left_on=col1_lower,
        right_on=col2_lower,
        how='right',
        suffixes=('_ref', '_other')
    )

    return matches[matches[col_df_1_for_comparison].notnull()].drop(columns=[col1_lower, col2_lower])

def not_exact_match(reference_df, df2, col_df_1_for_comparison, col_df_1_for_labels, col_df_2):
    """
    Not Exact Match Function

    This function compares two DataFrames to identify non-matching rows based on a specified column from a reference DataFrame and a column from another DataFrame. It performs a case-insensitive comparison by converting the relevant columns to lowercase.

    Parameters:
        reference_df (pd.DataFrame): The reference DataFrame containing the primary data for comparison.
        df2 (pd.DataFrame): The secondary DataFrame to be compared against the reference DataFrame.
        col_df_1_for_comparison (str): The column name in the reference DataFrame to be used for comparison.
        col_df_1_for_labels (str): The column name in the reference DataFrame to be used for labeling the results.
        col_df_2 (str): The column name in the secondary DataFrame to be compared.

    Returns:
        pd.DataFrame: A DataFrame containing the non-matching rows from the secondary DataFrame, excluding the comparison columns.
    """
    # create copies and prepare lowercase comparison columns
    df_1_trimmed = reference_df[[col_df_1_for_comparison, col_df_1_for_labels]].drop_duplicates().copy()
    df_2_trimmed = df2.copy()

    col1_lower = f'{col_df_1_for_comparison}_lower'
    col2_lower = f'{col_df_2}_lower'

    df_1_trimmed[col1_lower] = df_1_trimmed[col_df_1_for_comparison].str.lower()
    df_2_trimmed[col2_lower] = df_2_trimmed[col_df_2].str.lower()

    # merge with indicator and keep only the non-matching rows from either side
    matches = df_1_trimmed.merge(
        df_2_trimmed,
        left_on=col1_lower,
        right_on=col2_lower,
        how='right',
        suffixes=('_ref', '_other'),
        indicator=True
    )

    return matches[matches[col_df_1_for_comparison].isnull()].drop(columns=[col1_lower, col2_lower])

def best_match_jw(name, list_to_match):
    """
    Find the best matching supplier for a given name using Jaro-Winkler similarity.

    Parameters:
        name (str): The name to match against the supplier list.
        supplier_list (list of str): A list of supplier names to compare against.

    Returns:
        tuple: A tuple containing the best matching supplier (str) and the best score (float).
               If no match is found, returns (None, -1).
    """
    best_match = None
    best_score = -1
    for supplier in list_to_match:
        score = get_jaro_winkler_similarity(name, supplier)
        if score > best_score:
            best_score = score
            best_match = supplier
    return best_match, best_score

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
