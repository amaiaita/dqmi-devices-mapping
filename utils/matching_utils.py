import pandas as pd
import numpy as np
import re
from pyjarowinkler.distance import get_jaro_winkler_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def remove_tokens(text, tokens):
        tokens_set = set(tokens)
        words = text.split()
        cleaned_words = [w for w in words if w not in tokens_set]
        return " ".join(cleaned_words)

def clean_data(df: pd.DataFrame,
                select_col: str,
                token_col: str,
                tokens_to_remove: list,
                split_numbers: bool = True):
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

     # remove space after "non"
    df_output[token_col] = df_output[token_col].str.replace(
        r'non\s+',
        'non',
        regex=True
    )

    # collapse sequences of single-letter tokens: e.g., "w l a" -> "wla"
    df_output[token_col] = df_output[token_col].str.replace(
        r'\b(?:[a-z]\s+){1,}[a-z]\b',
        lambda m: m.group(0).replace(" ", ""),
        regex=True
    )
    if split_numbers:
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


def exact_match(reference_df, df2, col_df_1_for_comparison, col_df_1_for_labels, col_df_2, col_df2_label, device_match=False):
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
    )#.astype("Int64")
    l = ''
    if device_match:
        l = 'device_'
    matches[f'{l}level'] = np.where(matches[col_df_1_for_labels].notnull(), 'exact_match_' + col_df_1_for_comparison, None)

    df_devices = pd.concat([matches, df_2_prelabelled]).drop(columns = reference_df_cols)

    return df_devices

def exact_match_with_supplier_filter(reference_df, df2, col_df_1_for_comparison, col_df_1_for_labels, col_df_2, col_df2_label, df_2_supplier_col, df_reference_supplier_col):
    # create copies and prepare lowercase comparison columns
    
    if col_df2_label not in df2.columns:
        raise Exception("Label column does not align to existing label column")

    df_1_trimmed = reference_df.drop_duplicates(col_df_1_for_comparison)[[col_df_1_for_comparison, col_df_1_for_labels, df_reference_supplier_col]].copy()
    df_2_trimmed = df2[df2[col_df2_label].isnull()].copy()
    df_2_prelabelled = df2[df2[col_df2_label].notnull()].copy()
    matches = df_1_trimmed.merge(
        df_2_trimmed,
        left_on=[col_df_1_for_comparison, df_reference_supplier_col],
        right_on=[col_df_2, df_2_supplier_col],
        how='right',
    )

    matches[col_df2_label] = matches[f'{col_df_1_for_labels}'].where(
        matches[col_df_1_for_labels].notnull()
    )#.astype("Int64")

    matches['device_level'] = np.where(matches[col_df_1_for_labels].notnull(), 'exact_match_supplier_and_' + col_df_1_for_comparison, None)

    df_devices = pd.concat([matches, df_2_prelabelled]).drop(columns = ['NPC'])

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
    # df_to_match_trimmed = df_to_match[(df_to_match[label_col_name].isnull()) & (df_to_match['Manufacturer_label'].notnull())].copy()
    # df_2_prelabelled = df_to_match[df_to_match[label_col_name].notnull() | (df_to_match['Manufacturer_label'].isnull())].copy()
    df_to_match.astype({col_to_match: "str"}).dtypes
    df_to_match_trimmed = df_to_match[(df_to_match[label_col_name].isnull())].copy()
    df_2_prelabelled = df_to_match[df_to_match[label_col_name].notnull()].copy()
    # prep data
    df_reference_for_codes = df_reference.copy()
    df_reference_for_codes['NCP_starts'] = df_reference_for_codes['NPC'].str[:3]
    NPC_prefixes = df_reference_for_codes['NCP_starts'].unique().tolist()
    NPC_codes = df_reference_for_codes['NPC'].unique().tolist()
    NPC_codes = [code for code in NPC_codes if len(code) >= 5]
    logger.info("Device code level matching with {} column: {} unique NPC prefixes, {} unique NPC codes", col_to_match, len(NPC_prefixes), len(NPC_codes))
    
    MPC_codes = df_reference_for_codes['MPC'].unique().tolist()
    MPC_codes = [code for code in MPC_codes if len(code) >= 5]
    logger.info("Device code level matching with {} column: {} unique MPC codes", col_to_match, len(MPC_codes))
    EAN_codes = df_reference_for_codes['EAN/GTIN'].astype(str).unique().tolist()
    EAN_codes = [code for code in EAN_codes if len(code) >= 5]
    logger.info("Device code level matching with {} column: {} unique EAN/GTIN codes", col_to_match, len(EAN_codes))

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
        lambda x: pd.Series(detect_device_code_preference(logger, x, df_reference_for_codes, NPC_codes, npc_pattern, mpc_pattern, ean_pattern, col_to_match))
    )

    df_devices = pd.concat([df_to_match_trimmed, df_2_prelabelled])

    return df_devices

def detect_device_code_preference(logger, name, df_reference_for_codes, NPC_codes, npc_pattern, mpc_pattern, ean_pattern, col_to_match):
    failed_codes = None
    name = str(name)
    match = npc_pattern.search(name)
    if match:
        code = match.group(0).replace("-", "").replace(" ", "")
        if code in NPC_codes:
            return code, f'npc_code_match_{col_to_match}', None
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
        if len(npc_candidates) > 1:
            logger.warning("MPC code '{}' corresponds to multiple NPC codes", normalized_code)
            failed_codes = normalized_code
        elif len(npc_candidates) == 1:
            return npc_candidates.iloc[0], f'mpc_code_match_{col_to_match}', None
    
    # --- EAN/GTIN codes ---
    match = ean_pattern.search(name)
    if match:
        raw_code = match.group(0)
        normalized_code = raw_code.replace("-", "").replace(" ", "")
        npc_candidates = df_reference_for_codes.loc[
            df_reference_for_codes['EAN/GTIN'] == normalized_code, 'NPC'
        ]
        if len(npc_candidates) > 1:
            logger.warning("EAN/GTIN code '{}' corresponds to multiple NPC codes", normalized_code)
            failed_codes = normalized_code
        elif len(npc_candidates) == 1:
            return npc_candidates.iloc[0], f'ean_code_match_{col_to_match}', None
    
    return None, None, failed_codes

def bag_of_words_matching(df_catalogue, df_to_match, catalogue_tokens_col, to_match_tokens_col, score_threshold=0.7):
    """Match devices using TF-IDF bag of words similarity."""
    df_pre_matched = df_to_match[df_to_match['matched_device'].notnull()].copy()
    df_to_match = df_to_match[df_to_match['matched_device'].isnull()].copy()
    # Drop temporary columns from previous iterations (if they exist) to avoid duplicates
    cols_to_drop = ['best_similarity', 'n_best_matches', 'best_match_devices']
    df_to_match = df_to_match.drop(columns=cols_to_drop, errors='ignore')
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )
    corpus = pd.concat(
        [df_catalogue[catalogue_tokens_col],
            df_to_match[to_match_tokens_col]]
    )
    vectorizer.fit(corpus)
    x_to_match = vectorizer.transform(df_to_match[to_match_tokens_col])
    x_catalogue = vectorizer.transform(
        df_catalogue[catalogue_tokens_col]
    )
    similarity_matrix = cosine_similarity(x_to_match, x_catalogue)

    catalogue_idx_by_supplier = defaultdict(set)

    for i, supplier in enumerate(df_catalogue["Supplier"]):
        if pd.notna(supplier):
            catalogue_idx_by_supplier[supplier].add(i)
    rows = []
    all_catalogue_idxs = set(range(len(df_catalogue)))

    for i, supplier in enumerate(df_to_match["Supplier"]):
        if pd.notna(supplier) and supplier in catalogue_idx_by_supplier:
            valid_idxs = catalogue_idx_by_supplier[supplier]
        else:
            valid_idxs = all_catalogue_idxs

        row = similarity_matrix[i]

        masked_row = np.full_like(row, -1.0)
        masked_row[list(valid_idxs)] = row[list(valid_idxs)]

        max_score = masked_row.max()
        if max_score > score_threshold:
            match_idxs = np.where(masked_row == max_score)[0]
            best_devices = df_catalogue.iloc[match_idxs]["NPC"].tolist()
        else:
            match_idxs = []
            best_devices = []

        rows.append({
            "best_similarity": max_score,
            "n_best_matches": len(match_idxs),
            "best_match_devices": best_devices
        })
    df_devices = pd.concat(
        [df_to_match.reset_index(drop=True), pd.DataFrame(rows)],
        axis=1
    )
    matched = df_devices[
        (df_devices['best_similarity'] > score_threshold) &
        (df_devices['n_best_matches'] == 1)
    ]
    unmatched = df_devices[
        ~((df_devices['best_similarity'] > score_threshold) &
            (df_devices['n_best_matches'] == 1))
    ]
    unmatched['matched_device'] = None
    matched['matched_device'] = matched['best_match_devices'].str[0]
    matched['device_level'] = f'bag_of_words_match_{score_threshold}'
    df_devices = pd.concat([matched, unmatched, df_pre_matched])
    return df_devices

def bag_of_words_supplier_matching(df_to_match, df_reference, label_col_name, score_col_name, col_to_match, reference_col, reference_labels_col, level_col):
    """
    TF-IDF (unigrams+bigrams) + cosine similarity matching.

    Computes best similarity between `col_to_match` and `reference_col`, stores
    the score in `score_col_name`, and assigns the label from
    `reference_labels_col` to `label_col_name` when score > 0.7. Sets
    `level_col` for successful matches. Pre-matched rows are preserved.

    Parameters
    ----------
    df_to_match, df_reference : pd.DataFrame
    label_col_name, score_col_name, col_to_match, reference_col, reference_labels_col, level_col : str

    Returns
    -------
    pd.DataFrame
        Updated `df_to_match` with scores and assigned labels for matches.
    """
    df_pre_matched = df_to_match[df_to_match[label_col_name].notnull()].copy()
    df_to_match = df_to_match[df_to_match[label_col_name].isnull()].copy()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams help a lot
        stop_words="english"
    )

    # Fit on combined corpus so vectors are comparable
    corpus = pd.concat([df_to_match[col_to_match], df_reference[reference_col]])
    vectorizer.fit(corpus)

    X_a = vectorizer.transform(df_to_match[col_to_match])
    X_b = vectorizer.transform(df_reference[reference_col])
    similarity_matrix = cosine_similarity(X_a, X_b)
    best_match_idx = similarity_matrix.argmax(axis=1)
    best_match_score = similarity_matrix.max(axis=1)

    df_to_match[score_col_name] = best_match_score
    passed = best_match_score > 0.7
    df_to_match.loc[passed, label_col_name] = (
            df_reference.iloc[best_match_idx[passed]][reference_labels_col].values
        )
    
    df_matches = df_to_match[df_to_match[score_col_name] > 0.7]
    df_matches[level_col] = 'bag_of_words_match_' + reference_col
    df_rest = df_to_match[df_to_match[score_col_name] <= 0.7]
    
    df_devices = pd.concat([df_matches, df_rest, df_pre_matched])
    return df_devices

def substring_match(df_to_match, df_reference, label_col_name, col_to_match, reference_col, reference_labels_col, level_col):
    """
    Match rows by checking if reference text in df_reference[reference_col] is contained 
    within text of df_to_match[col_to_match].
    
    Parameters
    ----------
    df_to_match, df_reference : pd.DataFrame
    label_col_name, col_to_match, reference_col, reference_labels_col, level_col : str
    
    Returns
    -------
    pd.DataFrame
        Updated df_to_match with labels assigned where substring matches found.
    """
    df_pre_matched = df_to_match[df_to_match[label_col_name].notnull()].copy()
    df_to_match = df_to_match[df_to_match[label_col_name].isnull()].copy()
    
    reference_dict = df_reference.set_index(reference_col)[reference_labels_col].to_dict()
    reference_texts = list(reference_dict.keys())

    
    df_to_match[label_col_name] = df_to_match[col_to_match].apply(
        lambda x: find_substring_match(x, reference_texts, reference_dict)
    )
    
    df_matches = df_to_match[df_to_match[label_col_name].notnull()]
    df_matches[level_col] = 'substring_match_' + reference_col
    df_rest = df_to_match[df_to_match[label_col_name].isnull()]
    
    df_devices = pd.concat([df_matches, df_rest, df_pre_matched])
    return df_devices
    
def find_substring_match(text, reference_texts, reference_dict):
    text = str(text).lower().strip()
    if not text:
        return None
    for ref_text in reference_texts:
        ref_text_lower = str(ref_text).lower()
        if ref_text_lower in text:
            return reference_dict[ref_text]
    return None