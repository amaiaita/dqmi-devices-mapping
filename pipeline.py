from loguru import logger
import pandas as pd
from utils.matching_utils import clean_data, exact_match, not_exact_match, best_match_jw, number_of_tokens_match

logger.add("pipeline.log")

logger.info("Starting data cleaning pipeline")

# load catalogue data 
df_catalogue = pd.read_excel('data/Devices_catalogue.xlsx')
logger.info("Catalogue data loaded successfully. Number of records: {}", len(df_catalogue))

# load devices data
df_devices_data = pd.read_excel('data/raw_data_and_maps.xlsx')
logger.info("Devices data loaded successfully. Number of records: {}", len(df_devices_data))

# clean data and extract distinct tokens for manufacturers in both datasets
df_catalogue_supplier_distinct_tokens = clean_data(df_catalogue, 'Supplier', 'Supplier_tokens')
df_devices_data_distinct_tokens = clean_data(df_devices_data, 'CLN_Manufacturer', 'CLN_Manufacturer_tokens')

# remove nulls and empty strings from devices data - this is to avoid matching rates looking artificially low
df_devices_data_distinct_tokens = df_devices_data_distinct_tokens[
    df_devices_data_distinct_tokens['CLN_Manufacturer_tokens'].notnull() &
    (df_devices_data_distinct_tokens['CLN_Manufacturer_tokens'] != '')
]
logger.info("Null values removed from devices data. Remaining records: {}", len(df_devices_data_distinct_tokens))

logger.info("Starting supplier column matching process")

#  get rows where the supplier matches exactly based on tokens after cleaning
df_exact_matches_supplier = exact_match(df_catalogue_supplier_distinct_tokens, df_devices_data_distinct_tokens, 'Supplier_tokens', 'Supplier', 'CLN_Manufacturer_tokens')
logger.info("Exact matches found: {}", len(df_exact_matches_supplier))

# get remaining rows to match in later steps 
df_not_exact_matches_supplier = not_exact_match(df_catalogue_supplier_distinct_tokens, df_devices_data_distinct_tokens, 'Supplier_tokens', 'Supplier', 'CLN_Manufacturer_tokens').drop(['Supplier_tokens', 'Supplier'], axis='columns')
logger.info("Rows remaining to be matched: {}", len(df_not_exact_matches_supplier))

# create list of unique suppliers
unique_suppliers = df_catalogue_supplier_distinct_tokens['Supplier_tokens'].drop_duplicates().to_list()
logger.info("Unique suppliers extracted: {}", len(unique_suppliers))

# find best supplier matches using Jaro-Winkler similarity
df_not_exact_matches_supplier[['Supplier', 'Supplier_score']] = df_not_exact_matches_supplier['CLN_Manufacturer_tokens'].apply(
    lambda x: pd.Series(best_match_jw(x, unique_suppliers))
)

# filter matches with a score above the threshold
df_jw_match = df_not_exact_matches_supplier[df_not_exact_matches_supplier['Supplier_score'] >= 0.86]
logger.info("Jaro-Winkler matches found above threshold: {}", len(df_jw_match))

df_jw_not_match = df_not_exact_matches_supplier[df_not_exact_matches_supplier['Supplier_score'] < 0.86]
logger.info("Remaining to be matched: {}", len(df_jw_not_match))


df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_Manufacturer_tokens'].str.split()
df_catalogue_supplier_distinct_tokens['Supplier_tokens_list'] = df_catalogue_supplier_distinct_tokens['Supplier_tokens'].str.split()
supplier_tokens_list = df_catalogue_supplier_distinct_tokens['Supplier_tokens_list'].drop_duplicates().to_list()

# remove most common tokens that would inflate the percentage of overlap
cleaned_supplier_tokens_list = {}
tokens_to_remove = ['ltd', 'limited', 'uk', 'medical', 'new', 'international', 'formerly', 'in', 'nhs', 'technology', 'technologies', 'hcted']
for supplier in supplier_tokens_list:
    cleaned_supplier = []
    for token in supplier:
        if token not in tokens_to_remove:
            cleaned_supplier.append(token)
    cleaned_supplier_tokens_list[' '.join(supplier)] = cleaned_supplier

df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(lambda x: [token for token in x if token not in tokens_to_remove])

# find number of tokens match
df_jw_not_match[['Supplier', 'Supplier_score', 'Multiple_matches']] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(
    lambda x: pd.Series(number_of_tokens_match(x, cleaned_supplier_tokens_list))
)
# filter matches with more than 50% of tokens matching
df_tokens_match = df_jw_not_match[df_jw_not_match['Supplier_score'] > 0.5]
logger.info("Token overlaps matches found above threshold: {}", len(df_tokens_match))

# get final remaining unmatched rows
df_tokens_not_match = df_jw_not_match[df_jw_not_match['Supplier_score'] < 0.5]
logger.info("Final remaining unmatched records: {}", len(df_tokens_not_match))

logger.info("Percentage of rows that remain unmatched on the supplier field: {}%", (len(df_tokens_not_match) / len(df_devices_data)) * 100)

logger.info("Starting brand column matching process")

# attempting to continue on to match on the brand field
df_tokens_not_match = df_tokens_not_match.drop(['Multiple_matches', 'Supplier_score', 'CLN_manufacturer_tokens_list', '_merge'], axis='columns')
# clean data and extract distinct tokens for manufacturers in both datasets
df_catalogue_brand_distinct_tokens = clean_data(df_catalogue, 'Brand', 'Brand_tokens')

#  get rows where the brand matches exactly based on tokens after cleaning
df_exact_matches_brand = exact_match(df_catalogue_brand_distinct_tokens, df_tokens_not_match, 'Brand_tokens', 'Brand', 'CLN_Manufacturer_tokens')
logger.info("Exact matches found: {}", len(df_exact_matches_brand))

# get remaining rows to match in later steps 
df_not_exact_matches_brand = not_exact_match(df_catalogue_brand_distinct_tokens, df_tokens_not_match, 'Brand_tokens', 'Brand', 'CLN_Manufacturer_tokens').drop(['Brand_tokens', 'Brand'], axis='columns')
logger.info("Rows remaining to be matched: {}", len(df_not_exact_matches_brand))


# create list of unique brands
unique_brands = df_catalogue_brand_distinct_tokens['Brand_tokens'].drop_duplicates().to_list()
logger.info("Unique brands extracted: {}", len(unique_brands))

# find best supplier matches using Jaro-Winkler similarity
df_not_exact_matches_brand[['Brand', 'Brand_score']] = df_not_exact_matches_brand['CLN_Manufacturer_tokens'].apply(
    lambda x: pd.Series(best_match_jw(x, unique_brands))
)


# filter matches with a score above the threshold
df_jw_match = df_not_exact_matches_brand[df_not_exact_matches_brand['Brand_score'] >= 0.86]
logger.info("Jaro-Winkler matches found above threshold: {}", len(df_jw_match))

df_jw_not_match = df_not_exact_matches_brand[df_not_exact_matches_brand['Brand_score'] < 0.86]
logger.info("Remaining to be matched: {}", len(df_jw_not_match))


df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_Manufacturer_tokens'].str.split()
df_catalogue_brand_distinct_tokens['Brand_tokens_list'] = df_catalogue_brand_distinct_tokens['Brand_tokens'].str.split()
brand_tokens_list = df_catalogue_brand_distinct_tokens['Brand_tokens_list'].drop_duplicates().to_list()

cleaned_brand_tokens_list = {}
for brand in brand_tokens_list:
    cleaned_brand = []
    for token in brand:
        if token not in tokens_to_remove:
            cleaned_brand.append(token)
    cleaned_brand_tokens_list[' '.join(brand)] = cleaned_brand

df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(lambda x: [token for token in x if token not in tokens_to_remove])

# find number of tokens match
df_jw_not_match[['Brand', 'Brand_score', 'Multiple_matches']] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(
    lambda x: pd.Series(number_of_tokens_match(x, cleaned_brand_tokens_list))
)
# filter matches with more than 50% of tokens matching
df_tokens_match = df_jw_not_match[df_jw_not_match['Brand_score'] > 0.5]
logger.info("Token overlaps matches found above threshold: {}", len(df_tokens_match))

# get final remaining unmatched rows
df_tokens_not_match = df_jw_not_match[df_jw_not_match['Brand_score'] < 0.5]
logger.info("Final remaining unmatched records: {}", len(df_tokens_not_match))

logger.info("Percentage of rows that remain unmatched after brand matching: {}%", (len(df_tokens_not_match) / len(df_devices_data_distinct_tokens)) * 100)

logger.info("Data cleaning pipeline completed successfully.\n")