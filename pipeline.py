from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
from utils.matching_utils import clean_data, exact_match, best_match_jw, number_of_tokens_match

logger.add("logs/pipeline_{time}.log")

logger.info("Starting data cleaning pipeline")

# load catalogue data 
df_catalogue = pd.read_excel('data/Devices_catalogue.xlsx')
logger.info("Catalogue data loaded successfully. Number of records: {}", len(df_catalogue))

# keep only relevant columns for mapping + give them unique IDs to label with
df_catalogue_manufacturers = df_catalogue[['Supplier', 'Brand']].drop_duplicates()
df_catalogue_manufacturers = df_catalogue_manufacturers.reset_index()
df_catalogue_manufacturers.rename(columns={'index': 'catalogue_manufacturers_index'}, inplace=True)

# load devices data
df_devices_data = pd.read_excel('data/raw_data_and_maps.xlsx')
logger.info("Devices data loaded successfully. Number of records: {}", len(df_devices_data))

# keep only relevant columns for mapping + give them unique IDs to label with
df_devices = df_devices_data.reset_index()[['index', 'CLN_Manufacturer', 'CLN_Manufacturer_Device_Name']]
df_devices.to_csv('data/devices_to_map.csv', index=True)

# clean data and extract distinct tokens for manufacturers in both datasets
df_catalogue_suppliers = clean_data(df_catalogue_manufacturers, 'Supplier', 'Supplier_tokens')
df_devices = clean_data(df_devices, 'CLN_Manufacturer', 'CLN_Manufacturer_tokens')

# correctly label nulls as unmatchable
df_devices['Manufacturer_label'] = np.where((df_devices['CLN_Manufacturer_tokens'].isnull() |
    (df_devices['CLN_Manufacturer_tokens'] == '')), 'missing_manufacturer', None)
df_devices['level'] = np.where((df_devices['Manufacturer_label'] == 'missing_manufacturer'), 'null_level', None)

logger.info("Null values for manufacturer labelled. Remaining records to label: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

logger.info("Starting supplier column matching process")

#  get rows where the supplier matches exactly based on tokens after cleaning
df_devices = exact_match(df_catalogue_suppliers, df_devices, 'Supplier_tokens', 'catalogue_manufacturers_index', 'CLN_Manufacturer_tokens', 'Manufacturer_label')
logger.info("Exact matches found: {}", len(df_devices[df_devices['level']=='exact_match_Supplier_tokens']))
logger.info("Rows remaining to be matched: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

df_to_jw_match = df_devices[df_devices['Manufacturer_label'].isnull()]
df_matched_previously = df_devices[df_devices['Manufacturer_label'].notnull()]
# find best supplier matches using Jaro-Winkler similarity
df_to_jw_match[['Manufacturer_label', 'Supplier_score']] = df_to_jw_match['CLN_Manufacturer_tokens'].apply(
    lambda x: pd.Series(best_match_jw(x, df_catalogue_suppliers, 'Supplier_tokens', 'catalogue_manufacturers_index'))
)

# filter matches with a score above the threshold
df_jw_match = df_to_jw_match[df_to_jw_match['Supplier_score'] >= 0.86]
df_rest = df_to_jw_match[df_to_jw_match['Supplier_score'] < 0.86]

df_devices = pd.concat([df_matched_previously, df_jw_match, df_rest])

logger.info("Jaro-Winkler matches found above threshold: {}", len(df_jw_match))

logger.info("Remaining to be matched: {}", len(df_rest))


# df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_Manufacturer_tokens'].str.split()
# df_catalogue_supplier_distinct_tokens['Supplier_tokens_list'] = df_catalogue_supplier_distinct_tokens['Supplier_tokens'].str.split()
# supplier_tokens_list = df_catalogue_supplier_distinct_tokens['Supplier_tokens_list'].drop_duplicates().to_list()

# # remove most common tokens that would inflate the percentage of overlap
# cleaned_supplier_tokens_list = {}
# tokens_to_remove = ['ltd', 'limited', 'uk', 'medical', 'new', 'international', 'formerly', 'in', 'nhs', 'technology', 'technologies', 'hcted']
# for supplier in supplier_tokens_list:
#     cleaned_supplier = []
#     for token in supplier:
#         if token not in tokens_to_remove:
#             cleaned_supplier.append(token)
#     cleaned_supplier_tokens_list[' '.join(supplier)] = cleaned_supplier

# df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(lambda x: [token for token in x if token not in tokens_to_remove])

# # find number of tokens match
# df_jw_not_match[['Supplier', 'Supplier_score', 'Multiple_matches']] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(
#     lambda x: pd.Series(number_of_tokens_match(x, cleaned_supplier_tokens_list))
# )
# # filter matches with more than 50% of tokens matching
# df_tokens_match = df_jw_not_match[df_jw_not_match['Supplier_score'] > 0.5]
# logger.info("Token overlaps matches found above threshold: {}", len(df_tokens_match))

# # get final remaining unmatched rows
# df_tokens_not_match = df_jw_not_match[df_jw_not_match['Supplier_score'] < 0.5]
# logger.info("Final remaining unmatched records: {}", len(df_tokens_not_match))

# logger.info("Percentage of rows that remain unmatched on the supplier field: {:,.2f}%", (len(df_tokens_not_match) / len(df_devices_data)) * 100)

# logger.info("Starting brand column matching process")

# # attempting to continue on to match on the brand field
# df_tokens_not_match = df_tokens_not_match.drop(['Multiple_matches', 'Supplier_score', 'CLN_manufacturer_tokens_list', '_merge'], axis='columns')
# # clean data and extract distinct tokens for manufacturers in both datasets
# df_catalogue_brand_distinct_tokens = clean_data(df_catalogue, 'Brand', 'Brand_tokens')

# #  get rows where the brand matches exactly based on tokens after cleaning
# df_exact_matches_brand = exact_match(df_catalogue_brand_distinct_tokens, df_tokens_not_match, 'Brand_tokens', 'Brand', 'CLN_Manufacturer_tokens')
# logger.info("Exact matches found: {}", len(df_exact_matches_brand))

# # get remaining rows to match in later steps 
# df_not_exact_matches_brand = not_exact_match(df_catalogue_brand_distinct_tokens, df_tokens_not_match, 'Brand_tokens', 'Brand', 'CLN_Manufacturer_tokens').drop(['Brand_tokens', 'Brand'], axis='columns')
# logger.info("Rows remaining to be matched: {}", len(df_not_exact_matches_brand))


# # create list of unique brands
# unique_brands = df_catalogue_brand_distinct_tokens['Brand_tokens'].drop_duplicates().to_list()
# logger.info("Unique brands extracted: {}", len(unique_brands))

# # find best supplier matches using Jaro-Winkler similarity
# df_not_exact_matches_brand[['Brand', 'Brand_score']] = df_not_exact_matches_brand['CLN_Manufacturer_tokens'].apply(
#     lambda x: pd.Series(best_match_jw(x, unique_brands))
# )


# # filter matches with a score above the threshold
# df_jw_match = df_not_exact_matches_brand[df_not_exact_matches_brand['Brand_score'] >= 0.86]
# logger.info("Jaro-Winkler matches found above threshold: {}", len(df_jw_match))

# df_jw_not_match = df_not_exact_matches_brand[df_not_exact_matches_brand['Brand_score'] < 0.86]
# logger.info("Remaining to be matched: {}", len(df_jw_not_match))

# df_jw_not_match = df_jw_not_match.copy()
# df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_Manufacturer_tokens'].str.split()
# df_catalogue_brand_distinct_tokens['Brand_tokens_list'] = df_catalogue_brand_distinct_tokens['Brand_tokens'].str.split()
# brand_tokens_list = df_catalogue_brand_distinct_tokens['Brand_tokens_list'].drop_duplicates().to_list()

# cleaned_brand_tokens_list = {}
# for brand in brand_tokens_list:
#     cleaned_brand = []
#     for token in brand:
#         if token not in tokens_to_remove:
#             cleaned_brand.append(token)
#     cleaned_brand_tokens_list[' '.join(brand)] = cleaned_brand

# df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(lambda x: [token for token in x if token not in tokens_to_remove])

# # find number of tokens match
# df_jw_not_match[['Brand', 'Brand_score', 'Multiple_matches']] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(
#     lambda x: pd.Series(number_of_tokens_match(x, cleaned_brand_tokens_list))
# )
# # filter matches with more than 50% of tokens matching
# df_tokens_match = df_jw_not_match[df_jw_not_match['Brand_score'] > 0.5]
# logger.info("Token overlaps matches found above threshold: {}", len(df_tokens_match))

# # get final remaining unmatched rows
# df_tokens_not_match = df_jw_not_match[df_jw_not_match['Brand_score'] < 0.5]
# logger.info("Final remaining unmatched records: {}", len(df_tokens_not_match))

# logger.info("Percentage of rows that remain unmatched after brand matching: {:,.2f}%", (len(df_tokens_not_match) / len(df_devices_data_distinct_tokens)) * 100)

# logger.info("Starting matching process to remove suppliers not in the catalogue from eligible records")

# df_suppliers_not_in_catalogue = pd.read_csv('data/suppliers_not_in_catalogue.csv')

# # attempting to REMOVE SUPPLIERS NOT IN CATALOGUE
# df_tokens_not_match = df_tokens_not_match.drop(['Multiple_matches', 'Brand', 'Supplier','Brand_score', 'CLN_manufacturer_tokens_list', '_merge'], axis='columns')

# # clean data and extract distinct tokens for manufacturers in both datasets
# df_not_in_catalogue_suppliers_distinct_tokens = clean_data(df_suppliers_not_in_catalogue, 'Supplier_missing', 'Supplier_missing_tokens')

# #  get rows where the brand matches exactly based on tokens after cleaning
# df_exact_matches_brand = exact_match(df_not_in_catalogue_suppliers_distinct_tokens, df_tokens_not_match, 'Supplier_missing_tokens', 'Supplier_missing', 'CLN_Manufacturer_tokens')
# logger.info("Exact matches found: {}", len(df_exact_matches_brand))

# # get remaining rows to match in later steps 
# df_not_exact_matches_brand = not_exact_match(df_not_in_catalogue_suppliers_distinct_tokens, df_tokens_not_match, 'Supplier_missing_tokens', 'Supplier_missing', 'CLN_Manufacturer_tokens').drop(['Supplier_missing_tokens', 'Supplier_missing'], axis='columns')
# logger.info("Rows remaining to be matched: {}", len(df_not_exact_matches_brand))


# # create list of unique brands
# unique_brands = df_suppliers_not_in_catalogue

# # find best supplier matches using Jaro-Winkler similarity
# df_not_exact_matches_brand[['Supplier_missing', 'Supplier_missing_score']] = df_not_exact_matches_brand['CLN_Manufacturer_tokens'].apply(
#     lambda x: pd.Series(best_match_jw(x, unique_brands))
# )


# # filter matches with a score above the threshold
# df_jw_match = df_not_exact_matches_brand[df_not_exact_matches_brand['Supplier_missing_score'] >= 0.86]
# logger.info("Jaro-Winkler matches found above threshold: {}", len(df_jw_match))

# df_jw_not_match = df_not_exact_matches_brand[df_not_exact_matches_brand['Supplier_missing_score'] < 0.86]
# logger.info("Remaining to be matched: {}", len(df_jw_not_match))


# df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_Manufacturer_tokens'].str.split()
# df_not_in_catalogue_suppliers_distinct_tokens['Supplier_missing_tokens_list'] = df_not_in_catalogue_suppliers_distinct_tokens['Supplier_missing_tokens'].str.split()
# brand_tokens_list = df_not_in_catalogue_suppliers_distinct_tokens['Supplier_missing_tokens_list'].drop_duplicates().to_list()

# cleaned_brand_tokens_list = {}
# tokens_to_remove = ['ltd', 'limited', 'uk', 'medical', 'new', 'international', 'formerly', 'in', 'nhs', 'technology', 'technologies', 'hcted']
# for brand in brand_tokens_list:
#     cleaned_brand = []
#     for token in brand:
#         if token not in tokens_to_remove:
#             cleaned_brand.append(token)
#     cleaned_brand_tokens_list[' '.join(brand)] = cleaned_brand

# df_jw_not_match['CLN_manufacturer_tokens_list'] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(lambda x: [token for token in x if token not in tokens_to_remove])

# # find number of tokens match
# df_jw_not_match[['Supplier_missing', 'Supplier_missing_score', 'Multiple_matches']] = df_jw_not_match['CLN_manufacturer_tokens_list'].apply(
#     lambda x: pd.Series(number_of_tokens_match(x, cleaned_brand_tokens_list))
# )
# # filter matches with more than 50% of tokens matching
# df_tokens_match = df_jw_not_match[df_jw_not_match['Supplier_missing_score'] > 0.5]
# logger.info("Token overlaps matches found above threshold: {}", len(df_tokens_match))

# # get final remaining unmatched rows
# df_tokens_not_match = df_jw_not_match[df_jw_not_match['Supplier_missing_score'] < 0.5]
# logger.info("Final remaining unmatched records: {}", len(df_tokens_not_match))

# logger.info("Percentage of rows that remain unmatched: {:,.2f}%", (len(df_tokens_not_match) / len(df_devices_data_distinct_tokens)) * 100)

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = f'data/unmatched_rows_{timestamp}.csv'
# df_tokens_not_match.to_csv(output_path, index=False)
# logger.info("Unmatched rows saved to {}", output_path)

# logger.info("Matching Manufacturer pipeline completed successfully.\n")