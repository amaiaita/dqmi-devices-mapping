import pandas as pd
import numpy as np
from pyjarowinkler.distance import get_jaro_winkler_similarity

def remove_tokens(text, tokens):
        tokens_set = set(tokens)
        words = text.split()
        cleaned_words = [w for w in words if w not in tokens_set]
        return " ".join(cleaned_words)

def clean_data(df: pd.DataFrame,
                select_col: str,
                token_col: str,
                tokens_to_remove: list):
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

    df_output[token_col] = df_output[token_col].astype(str).str.lower().apply(lambda x: remove_tokens(x, tokens_to_remove))
    df_output[f'{token_col}_list'] = df_output[token_col].str.split()

    return df_output


def exact_match(reference_df, df2, col_df_1_for_comparison, col_df_1_for_labels, col_df_2, col_df2_label):
    # create copies and prepare lowercase comparison columns
    
    if col_df2_label not in df2.columns:
        raise Exception("Label column does not align to existing label column")

    df_1_trimmed = reference_df.drop_duplicates(col_df_1_for_comparison)[[col_df_1_for_comparison, col_df_1_for_labels]].copy()
    reference_df_cols = df_1_trimmed.columns.tolist()
    df_2_trimmed = df2[df2[col_df2_label].isnull()].copy()
    df_2_prelabelled = df2[df2[col_df2_label].notnull()].copy()

    matches = df_1_trimmed.merge(
        df_2_trimmed,
        left_on=col_df_1_for_comparison,
        right_on=col_df_2,
        how='right',
    )

    matches[col_df2_label] = matches[col_df_1_for_labels].where(
        matches[col_df_1_for_labels].notnull()
    ).astype("Int64")
    matches['level'] = np.where(matches[col_df_1_for_labels].notnull(), 'exact_match_' + col_df_1_for_comparison, None)
    print('exact_match_' + col_df_1_for_labels)
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
    reference_df_cols = df_reference.columns.tolist()
    # find best supplier matches using Jaro-Winkler similarity
    df_to_jw_match[[label_col_name, score_col_name]] = df_to_jw_match[col_to_match].apply(
        lambda x: pd.Series(best_match_jw(x, df_reference, reference_col, reference_labels_col))
    )

    # filter matches with a score above the threshold
    df_jw_match = df_to_jw_match[df_to_jw_match[score_col_name] >= score_threshold]
    df_rest = df_to_jw_match[df_to_jw_match[score_col_name] < score_threshold]
    df_jw_match['level'] = 'jaro_winkler_match_' + reference_col
    df_rest[label_col_name] = None

    logger.info("Jaro-Winkler matches found above threshold: {}", len(df_jw_match))
    logger.info("Remaining to be matched: {}", len(df_rest))

    reference_df_cols.extend([score_col_name])
    df_output = pd.concat([df_matched_previously, df_jw_match, df_rest]).drop(columns = reference_df_cols, errors='ignore')
    
    df_output[label_col_name] = df_output[label_col_name]
    # .astype("Int64")


    return df_output

def number_of_tokens_overlap(name_tokens, df_to_match, col_to_match, col_labels):
    """
    Compare a list of tokens (name_tokens) against each row in df_to_match,
    where df_to_match[col_to_match] contains a list of tokens.
    """

    # keep only the two necessary columns
    df = df_to_match[[col_to_match, col_labels]].copy()

    best_match = None
    best_score = 0
    multiple_matches = False
    best_match_list = []

    name_set = set(name_tokens)

    for idx, row in df.iterrows():
        supplier_tokens = row[col_to_match] or []  # ensure list
        supplier_set = set(supplier_tokens)

        if len(supplier_set) == 0:
            score = 0
        else:
            common_tokens = name_set & supplier_set
            score = len(common_tokens) / len(supplier_set)

        if score > best_score:
            best_score = score
            best_match = row[col_labels]
            best_match_list = [row[col_labels]]
            multiple_matches = False

        elif score == best_score and score != 0:
            multiple_matches = True
            best_match_list.append(row[col_labels])

    if multiple_matches:
        return best_match_list, 0, True

    return best_match, best_score, False



def number_of_tokens_match(logger, df_to_match, df_reference, label_col_name, score_col_name, col_to_match, reference_col, reference_labels_col, score_threshold=0.5):
    df_to_token_match = df_to_match[df_to_match[label_col_name].isnull()]
    df_matched_previously = df_to_match[df_to_match[label_col_name].notnull()]
    reference_df_cols = df_reference.columns.tolist()
    # find number of tokens match
    df_to_token_match[[label_col_name, score_col_name, 'Multiple_matches']] = df_to_token_match[col_to_match].apply(
        lambda x: pd.Series(number_of_tokens_overlap(x, df_reference, reference_col, reference_labels_col))
    )

    df_tokens_match = df_to_token_match[df_to_token_match[score_col_name] > score_threshold]
    df_tokens_match['level'] = 'token_overlap_match_' + col_to_match
    logger.info("Token overlaps matches found above threshold: {}", len(df_tokens_match))
    
    df_rest = df_to_token_match[df_to_token_match[score_col_name] <= score_threshold]
    df_rest[label_col_name] = None
    logger.info("Final remaining unmatched records: {}", len(df_rest))

    reference_df_cols.extend([score_col_name, 'Multiple_matches'])
    df_output = pd.concat([df_matched_previously, df_tokens_match, df_rest]).drop(columns = reference_df_cols, errors='ignore')
    return df_output
