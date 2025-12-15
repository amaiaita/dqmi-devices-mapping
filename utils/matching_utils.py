import pandas as pd
import numpy as np
import re
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

    return df_output

def number_of_tokens_overlap(name_tokens, df_to_match, col_to_match, col_labels):
    """
    Compare a list of tokens (name_tokens) against each row in df_to_match using vectorized operations.
    """
    df = df_to_match[[col_to_match, col_labels]].copy()
    name_set = set(name_tokens) if name_tokens else set()

    # vectorized: compute overlap score for all rows at once
    df['overlap_count'] = df[col_to_match].apply(
        lambda tokens: len(name_set & set(tokens)) if tokens else 0
    )
    df['supplier_count'] = df[col_to_match].apply(
        lambda tokens: len(tokens) if tokens else 1  # avoid div by zero
    )
    df['score'] = df['overlap_count'] / df['supplier_count']

    # find best score(s)
    best_score = df['score'].max() if len(df) > 0 else 0

    if best_score == 0:
        return None, 0, False

    best_matches = df[df['score'] == best_score]
    multiple_matches = len(best_matches) > 1
    best_match = best_matches[col_labels].iloc[0]
    best_match_list = best_matches[col_labels].tolist() if multiple_matches else [best_match]

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
    df_tokens_match['level'] = 'token_overlap_match_' + reference_col
    logger.info("Token overlaps matches found above threshold: {}", len(df_tokens_match))
    
    df_rest = df_to_token_match[df_to_token_match[score_col_name] <= score_threshold]
    df_rest[label_col_name] = None
    logger.info("Final remaining unmatched records: {}", len(df_rest))

    reference_df_cols.extend([score_col_name, 'Multiple_matches'])
    df_output = pd.concat([df_matched_previously, df_tokens_match, df_rest]).drop(columns = reference_df_cols, errors='ignore')
    return df_output

def device_code_level_matching(df_to_match, df_reference, label_col_name, col_to_match, logger=None):
    df_to_match_trimmed = df_to_match[(df_to_match[label_col_name].isnull()) & (df_to_match['Manufacturer_label'].notnull())].copy()
    df_2_prelabelled = df_to_match[df_to_match[label_col_name].notnull() | (df_to_match['Manufacturer_label'].isnull())].copy()
    
    # prep data
    df_reference_for_codes = df_reference.copy()
    df_reference_for_codes['NCP_starts'] = df_reference_for_codes['NPC'].str[:3]
    NPC_prefixes = df_reference_for_codes['NCP_starts'].unique().tolist()
    NPC_codes = df_reference_for_codes['NPC'].unique().tolist()
    logger.info("Device code level matching: {} unique NPC prefixes, {} unique NPC codes", len(NPC_prefixes), len(NPC_codes))
    
    MPC_codes = df_reference_for_codes['MPC'].unique().tolist()
    logger.info("Device code level matching: {} unique MPC codes", len(MPC_codes))
    
    EAN_codes = df_reference_for_codes['EAN/GTIN'].astype(str).unique().tolist()
    logger.info("Device code level matching: {} unique EAN/GTIN codes", len(EAN_codes))

    npc_pattern = re.compile(
        r"\b(" +
        "|".join([re.escape(p) + r'[- ]?\d+' for p in NPC_prefixes]) +
        r")\b",
        flags=re.IGNORECASE
    )

    mpc_pattern = re.compile(
        r"\b(" +
        "|".join([re.escape(code[:3]) + r'[- ]?' + re.escape(code[3:]) for code in MPC_codes]) +
        r")\b",
        flags=re.IGNORECASE
    )

    ean_pattern = re.compile(
        r"\b(" +
        "|".join([re.escape(code[:3]) + r'[- ]?' + re.escape(code[3:]) for code in EAN_codes]) +
        r")\b",
        flags=re.IGNORECASE
    )

    df_to_match_trimmed[[label_col_name, 'device_level', 'codes_failed_to_match']] = df_to_match_trimmed[col_to_match].apply(
        lambda x: pd.Series(detect_device_code_preference(logger, x, df_reference_for_codes, NPC_codes, npc_pattern, mpc_pattern, ean_pattern))
    )

    df_devices = pd.concat([df_to_match_trimmed, df_2_prelabelled])

    return df_devices

def detect_device_code_preference(logger, name, df_reference_for_codes, NPC_codes, npc_pattern, mpc_pattern, ean_pattern):
    # --- NPC CODES ---    
    failed_codes = None
    match = npc_pattern.search(name)
    if match:
        code = match.group(0).replace("-", "").replace(" ", "")
        if code in NPC_codes:
            return code, 'npc_code_match', None
        else:
            logger.warning("Detected NPC code '{}' not in reference list", match.group(0))
            failed_codes = code

    # --- MPC CODES ---    

    match = mpc_pattern.search(name)
    if match:
        raw_code = match.group(0)
        normalized_code = raw_code.replace("-", "").replace(" ", "")

        npc_candidates = df_reference_for_codes.loc[
            df_reference_for_codes['MPC'] == normalized_code, 'NPC'
        ]

        if npc_candidates.count() > 1:
            logger.warning("MPC code '{}' corresponds to multiple NPC codes", normalized_code)
            failed_codes = normalized_code
        elif npc_candidates.count() == 1:
            return npc_candidates.iloc[0], 'mpc_code_match', None
    
    # EAN/GTIN codes
    match = ean_pattern.search(name)
    if match:
        raw_code = match.group(0)
        normalized_code = raw_code.replace("-", "").replace(" ", "")

        npc_candidates = df_reference_for_codes.loc[
            df_reference_for_codes['EAN/GTIN'] == normalized_code, 'NPC'
        ]

        if npc_candidates.count() > 1:
            logger.warning("EAN/GTIN code '{}' corresponds to multiple NPC codes", normalized_code)
            failed_codes = normalized_code
        elif npc_candidates.count() == 1:
            return npc_candidates.iloc[0], 'ean_code_match', None
    return None, None, failed_codes
