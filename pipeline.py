from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
from utils.matching_utils import clean_data, exact_match, number_of_tokens_match, jaro_winkler_match

logger.add("logs/pipeline_{time}.log")

logger.info("Starting data cleaning pipeline")

# load catalogue data 
df_catalogue = pd.read_excel('data/Devices_catalogue.xlsx')
logger.info("Catalogue data loaded successfully. Number of records: {}", len(df_catalogue))

# keep only relevant columns for mapping + give them unique IDs to label with
df_catalogue_manufacturers = df_catalogue[['Supplier', 'Brand']].drop_duplicates()
df_catalogue_manufacturers = df_catalogue_manufacturers.reset_index()
df_catalogue_manufacturers.rename(columns={'index': 'catalogue_manufacturers_index'}, inplace=True)

print(df_catalogue_manufacturers.columns)

# load devices data
df_devices_data = pd.read_excel('data/raw_data_and_maps.xlsx')
logger.info("Devices data loaded successfully. Number of records: {}", len(df_devices_data))

# keep only relevant columns for mapping + give them unique IDs to label with
df_devices = df_devices_data.reset_index()[['index', 'CLN_Manufacturer', 'CLN_Manufacturer_Device_Name']]
df_devices.to_csv('data/devices_to_map.csv', index=True)

tokens_to_remove = ['ltd', 'limited', 'uk', 'medical', 'new', 'international', 'formerly', 'in', 'nhs', 'technology', 'technologies', 'hcted', 'e direct']

# clean data and extract distinct tokens for manufacturers in both datasets
df_catalogue_suppliers = clean_data(df_catalogue_manufacturers, 'Supplier', 'Supplier_tokens', tokens_to_remove)
df_devices = clean_data(df_devices, 'CLN_Manufacturer', 'CLN_Manufacturer_tokens', tokens_to_remove)

# # correctly label nulls as unmatchable
df_devices['Manufacturer_label'] = np.where((df_devices['CLN_Manufacturer_tokens'].isnull() |
    (df_devices['CLN_Manufacturer_tokens'] == '')), 'missing_manufacturer', None)
df_devices['level'] = np.where((df_devices['Manufacturer_label'] == 'missing_manufacturer'), 'null_level', None)

logger.info("Null values for manufacturer labelled. Remaining records to label: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

logger.info("Starting supplier column matching process")

#  get rows where the supplier matches exactly based on tokens after cleaning
df_devices = exact_match(df_catalogue_suppliers, df_devices, 'Supplier_tokens', 'catalogue_manufacturers_index', 'CLN_Manufacturer_tokens', 'Manufacturer_label')
logger.info("Exact matches found: {}", len(df_devices[df_devices['level']=='exact_match_Supplier_tokens']))
logger.info("Rows remaining to be matched: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

df_devices = jaro_winkler_match(logger, df_devices, df_catalogue_suppliers, 'Manufacturer_label', 'Supplier_score', 'CLN_Manufacturer_tokens', 'Supplier_tokens','catalogue_manufacturers_index', 0.86)

df_devices = number_of_tokens_match(logger, df_devices, df_catalogue_suppliers, 'Manufacturer_label', 'Supplier_score', 'CLN_Manufacturer_tokens_list', 'Supplier_tokens_list', 'catalogue_manufacturers_index', 0.5)

logger.info("Percentage of rows that remain unmatched on the supplier field: {:,.2f}%", (len(df_devices[df_devices['Manufacturer_label'].isnull()])/len(df_devices))*100)

logger.info("Starting brand column matching process")

# attempting to continue on to match on the brand field

#  get rows where the brand matches exactly based on tokens after cleaning
df_catalogue_brand = clean_data(df_catalogue_manufacturers, 'Brand', 'Brand_tokens', tokens_to_remove)

df_devices = exact_match(df_catalogue_brand, df_devices, 'Brand_tokens', 'catalogue_manufacturers_index', 'CLN_Manufacturer_tokens', 'Manufacturer_label')
logger.info("Exact matches found: {}", len(df_devices[df_devices['level']=='exact_match_Brand_tokens']))
logger.info("Rows remaining to be matched: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

df_devices = jaro_winkler_match(logger, df_devices, df_catalogue_brand, 'Manufacturer_label', 'Brand_score', 'CLN_Manufacturer_tokens', 'Brand_tokens','catalogue_manufacturers_index', 0.86)
logger.info("Jaro Winkler matches found: {}", len(df_devices[df_devices['level']=='jaro_winkler_Brand_tokens']))
logger.info("Rows remaining to be matched: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

df_devices = number_of_tokens_match(logger, df_devices, df_catalogue_brand, 'Manufacturer_label', 'Brand_score', 'CLN_Manufacturer_tokens_list', 'Brand_tokens_list', 'catalogue_manufacturers_index', 0.5)

logger.info("Percentage of rows that remain unmatched on the brand field: {:,.2f}%", (len(df_devices[df_devices['Manufacturer_label'].isnull()])/len(df_devices))*100)

logger.info("Starting matching process to remove suppliers not in the catalogue from eligible records")

df_suppliers_not_in_catalogue = pd.read_csv('data/suppliers_not_in_catalogue.csv')
logger.info("Suppliers not in catalogue data loaded successfully. Number of records: {}", len(df_suppliers_not_in_catalogue))

df_catalogue_suppliers_not_in_catalogue = clean_data(df_suppliers_not_in_catalogue, 'Supplier_missing', 'Supplier_missing_tokens', tokens_to_remove)
df_catalogue_suppliers_not_in_catalogue['catalogue_manufacturers_index'] = -1
logger.info("Starting suppliers not in catalogue column matching process")

#  get rows where the supplier matches exactly based on tokens after cleaning
df_devices = exact_match(df_catalogue_suppliers_not_in_catalogue, df_devices, 'Supplier_missing_tokens', 'catalogue_manufacturers_index', 'CLN_Manufacturer_tokens', 'Manufacturer_label')
logger.info("Exact matches found: {}", len(df_devices[df_devices['level']=='exact_match_Supplier_missing_tokens']))
logger.info("Rows remaining to be matched: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

df_devices = jaro_winkler_match(logger, df_devices, df_catalogue_suppliers_not_in_catalogue, 'Manufacturer_label', 'Supplier_missing_score', 'CLN_Manufacturer_tokens', 'Supplier_missing_tokens','catalogue_manufacturers_index', 0.86)

df_devices = number_of_tokens_match(logger, df_devices, df_catalogue_suppliers_not_in_catalogue, 'Manufacturer_label', 'Supplier_missing_score', 'CLN_Manufacturer_tokens_list', 'Supplier_missing_tokens_list', 'catalogue_manufacturers_index', 0.5)

logger.info("Percentage of rows that remain unmatched on the supplier field: {:,.2f}%", (len(df_devices[df_devices['Manufacturer_label'].isnull()])/len(df_devices))*100)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f'data/outputs/matching_results{timestamp}.csv'
df_devices.to_csv(output_path, index=False)
logger.info("Unmatched rows saved to {}", output_path)

logger.info("Matching Manufacturer pipeline completed successfully.\n")