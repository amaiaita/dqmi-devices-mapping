import os
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
from utils.matching_utils import clean_data, exact_match, number_of_tokens_match, jaro_winkler_match, device_code_level_matching, bag_of_words_matching, exact_match_with_supplier_filter, bag_of_words_supplier_matching, substring_match, catalogue_manufacturer_in_device_name_matching

run_manufacturer_mapping = True
run_device_name_mapping = False
manufacturer_mapping_file = ''
create_log_file = True
jw_threshold = 0.9

if create_log_file:
    logger.add("logs/pipeline_{time}.log")

if run_manufacturer_mapping:
    logger.info("Starting data cleaning pipeline")

    # load catalogue data 
    df_catalogue = pd.read_excel('data/Devices_catalogue.xlsx')
    logger.info("Catalogue data loaded successfully. Number of records: {}", len(df_catalogue))

    # keep only relevant columns for mapping + give them unique IDs to label with
    df_catalogue_manufacturers = df_catalogue[['Supplier', 'Brand']].drop_duplicates()
    df_catalogue_manufacturers = df_catalogue_manufacturers.reset_index()
    df_catalogue_manufacturers.rename(columns={'index': 'catalogue_manufacturers_index'}, inplace=True)
    df_catalogue_manufacturers.to_csv('data/catalogue_with_index.csv')

    # load devices data
    df_devices_data = pd.read_excel('data/raw_data_and_maps.xlsx')
    logger.info("Devices data loaded successfully. Number of records: {}", len(df_devices_data))

    df_devices = df_devices_data.reset_index()[['index', 'CLN_Manufacturer', 'CLN_Manufacturer_Device_Name', 'CLN_Device_Serial_Number']]
    df_devices.to_csv('data/devices_to_map.csv', index=True)

    tokens_to_remove = ['ltd', 'limited', 'uk', 'medical', 'new', 'international', 'formerly', 'in', 'nhs', 'technology', 'technologies', 'hcted', 'e direct', 'gmbh', 'europe', 'needle', 'accessories']

    # clean data and extract distinct tokens for manufacturers in both datasets
    df_catalogue_suppliers = clean_data(df_catalogue_manufacturers, 'Supplier', 'Supplier_tokens', tokens_to_remove)
    df_devices = clean_data(df_devices, 'CLN_Manufacturer', 'CLN_Manufacturer_tokens', tokens_to_remove)

    # # correctly label nulls as unmatchable
    df_devices['Manufacturer_label'] = np.where((df_devices['CLN_Manufacturer_tokens'].isnull() |
    (df_devices['CLN_Manufacturer_tokens'] == '')), -99, None)
    df_devices['level'] = np.where((df_devices['Manufacturer_label'] == -99), 'null_level', None)

    logger.info("Null values for manufacturer labelled. Remaining records to label: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

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

    df_devices = jaro_winkler_match(logger, df_devices, df_catalogue_suppliers_not_in_catalogue, 'Manufacturer_label', 'Supplier_missing_score', 'CLN_Manufacturer_tokens', 'Supplier_missing_tokens','catalogue_manufacturers_index', jw_threshold)

    df_devices = number_of_tokens_match(logger, df_devices, df_catalogue_suppliers_not_in_catalogue, 'Manufacturer_label', 'Supplier_missing_score', 'CLN_Manufacturer_tokens_list', 'Supplier_missing_tokens_list', 'catalogue_manufacturers_index', 0.5)
    df_devices = catalogue_manufacturer_in_device_name_matching(df_devices, df_catalogue_suppliers_not_in_catalogue, 'Manufacturer_label', 'CLN_Manufacturer_Device_Name', 'Supplier_missing_tokens', logger)

    logger.info("Percentage of rows that remain unmatched on the not in catalogue field: {:,.2f}%", (len(df_devices[df_devices['Manufacturer_label'].isnull()])/len(df_devices))*100)

    logger.info("Starting supplier column matching process")

    #  get rows where the supplier matches exactly based on tokens after cleaning
    df_devices = exact_match(df_catalogue_suppliers, df_devices, 'Supplier_tokens', 'catalogue_manufacturers_index', 'CLN_Manufacturer_tokens', 'Manufacturer_label')
    logger.info("Exact matches found: {}", len(df_devices[df_devices['level']=='exact_match_Supplier_tokens']))
    logger.info("Rows remaining to be matched: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))
    
    df_devices = substring_match(df_devices, df_catalogue_suppliers, 'Manufacturer_label', 'CLN_Manufacturer_tokens','Supplier_tokens', 'catalogue_manufacturers_index', 'level')
    logger.info("Substring matches found: {}", len(df_devices[df_devices['level']=='substring_match_Supplier_tokens']))
    df_devices = jaro_winkler_match(logger, df_devices, df_catalogue_suppliers, 'Manufacturer_label', 'Supplier_score', 'CLN_Manufacturer_tokens', 'Supplier_tokens','catalogue_manufacturers_index', jw_threshold)

    df_devices = number_of_tokens_match(logger, df_devices, df_catalogue_suppliers, 'Manufacturer_label', 'Supplier_score', 'CLN_Manufacturer_tokens_list', 'Supplier_tokens_list', 'catalogue_manufacturers_index', 0.5)

    df_devices = bag_of_words_supplier_matching(df_devices, df_catalogue_suppliers, 'Manufacturer_label', 'Supplier_score', 'CLN_Manufacturer_tokens', 'Supplier_tokens', 'catalogue_manufacturers_index', 'level')
    logger.info("Bag of words matches found: {}", len(df_devices[df_devices['level']=='bag_of_words_match_Supplier_tokens']))
    logger.info("Percentage of rows that remain unmatched on the supplier field: {:,.2f}%", (len(df_devices[df_devices['Manufacturer_label'].isnull()])/len(df_devices))*100)

    df_devices = catalogue_manufacturer_in_device_name_matching(df_devices, df_catalogue_suppliers, 'Manufacturer_label', 'CLN_Manufacturer_Device_Name', 'Supplier_tokens', logger)

    logger.info("Starting brand column matching process")

    # attempting to continue on to match on the brand field

    #  get rows where the brand matches exactly based on tokens after cleaning
    df_catalogue_brand = clean_data(df_catalogue_manufacturers, 'Brand', 'Brand_tokens', tokens_to_remove)

    df_devices = exact_match(df_catalogue_brand, df_devices, 'Brand_tokens', 'catalogue_manufacturers_index', 'CLN_Manufacturer_tokens', 'Manufacturer_label')
    logger.info("Exact matches found: {}", len(df_devices[df_devices['level']=='exact_match_Brand_tokens']))
    logger.info("Rows remaining to be matched: {}", len(df_devices[df_devices['Manufacturer_label'].isnull()]))

    df_devices = jaro_winkler_match(logger, df_devices, df_catalogue_brand, 'Manufacturer_label', 'Brand_score', 'CLN_Manufacturer_tokens', 'Brand_tokens','catalogue_manufacturers_index', jw_threshold)

    df_devices = number_of_tokens_match(logger, df_devices, df_catalogue_brand, 'Manufacturer_label', 'Brand_score', 'CLN_Manufacturer_tokens_list', 'Brand_tokens_list', 'catalogue_manufacturers_index', 0.5)
    
    df_devices = catalogue_manufacturer_in_device_name_matching(df_devices, df_catalogue_brand, 'Manufacturer_label', 'CLN_Manufacturer_Device_Name', 'Brand_tokens', logger)

    logger.info("Percentage of rows that remain unmatched on the brand field: {:,.2f}%", (len(df_devices[df_devices['Manufacturer_label'].isnull()])/len(df_devices))*100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'data/outputs/matching_results_manufacturer{timestamp}.csv'
    df_devices.to_csv(output_path, index=False)
    logger.info("Output saved to {}", output_path)

    logger.info("Matching Manufacturer pipeline completed successfully.\n")

if run_device_name_mapping:
    logger.info("Beginning of Device Name mapping pipeline.\n")

    directory = 'data/outputs'
    if manufacturer_mapping_file != '':
        path = f'{directory}/{manufacturer_mapping_file}'
        df_devices = pd.read_csv(f'{path}.csv')
    else:
        output_files = sorted(os.listdir(directory))
        path = f'{directory}/{output_files[-1]}'
        df_devices = pd.read_csv(f'{path}')

    logger.info("Device data loaded successfully from {}. Number of records: {}", path, len(df_devices))
    
    logger.info("Adding linked manufacturers to device input data.")
    df_catalogue = pd.read_csv('data/catalogue_with_index.csv',index_col='Unnamed: 0')
    df_catalogue['Manufacturer_label'] = df_catalogue['catalogue_manufacturers_index']
    df_devices['Manufacturer_label'] = pd.to_numeric(df_devices['Manufacturer_label'], errors='coerce').astype('Int64')
    df_catalogue['Manufacturer_label'] = df_catalogue['Manufacturer_label'].astype('Int64')
    df_devices = df_devices.merge(
        df_catalogue[['Manufacturer_label', 'Supplier', 'Brand']],
        on='Manufacturer_label',
        how='left',
        suffixes=('_dev', '_cat')
    )
    df_devices = df_devices[['CLN_Manufacturer', 'CLN_Manufacturer_Device_Name', 'CLN_Device_Serial_Number', 'Supplier', 'Manufacturer_label']]
    df_catalogue = pd.read_excel('data/Devices_catalogue.xlsx')

    df_devices['matched_device'] = np.where((df_devices['CLN_Manufacturer_Device_Name'].isnull() |
        (df_devices['CLN_Manufacturer_Device_Name'] == '')), 'null device', None)
    df_devices['device_level'] = np.where((df_devices['CLN_Manufacturer_Device_Name'] == -99), 'null_level', None)

    logger.info("Null values for device name labelled. Remaining records to label: {}", len(df_devices[df_devices['matched_device'].isnull()]))

    df_devices.loc[
        df_devices['Manufacturer_label'] == -1,
        'matched_device'
    ] = 'manufacturer not in catalogue'    
    df_devices.loc[
        df_devices['Manufacturer_label'] == -1,
        'device_level'
    ] = 'not_in_catalogue'

    logger.info("Values not in catalogue labelled. Remaining records to label: {}", len(df_devices[df_devices['matched_device'].isnull()]))
    
    device_tokens_to_remove = ['and', 'hcted']

    df_devices['device_manufacturer_concat'] = df_devices['Supplier'].astype(str) + ' ' + df_devices['CLN_Manufacturer_Device_Name'].astype(str)
    df_devices = clean_data(df_devices, 'device_manufacturer_concat', 'supplier_and_device_name_tokens', device_tokens_to_remove, split_numbers=False)
    df_catalogue['Device_concat'] = df_catalogue['Supplier'].astype(str) + ' ' + df_catalogue['Base Description'].astype(str) + ' ' + df_catalogue['Secondary Description'].astype(str)
    df_catalogue = clean_data(df_catalogue, 'Device_concat', 'catalogue_device_name_and_manufacturer_tokens', device_tokens_to_remove, split_numbers=False)

    df_devices = clean_data(df_devices, 'CLN_Manufacturer_Device_Name', 'device_name_tokens', device_tokens_to_remove, split_numbers=False)
    df_catalogue['Device_only_concat'] = df_catalogue['Base Description'].astype(str) + ' ' + df_catalogue['Secondary Description'].astype(str)
    df_catalogue = clean_data(df_catalogue, 'Device_only_concat', 'catalogue_device_only_tokens', device_tokens_to_remove, split_numbers=False)

    df_devices = exact_match(df_catalogue, df_devices, 'catalogue_device_name_and_manufacturer_tokens', 'NPC', 'supplier_and_device_name_tokens', 'matched_device', True)
    logger.info("Exact match level matching completed. Number of records labelled: {}", len(df_devices[df_devices['device_level']=='exact_match_catalogue_device_name_and_manufacturer_tokens']))

    df_devices = device_code_level_matching(df_devices, df_catalogue, 'matched_device', 'CLN_Device_Serial_Number', logger)
    logger.info("Device code level matching completed for serial number column. Number of records labelled at NPC code level: {}", len(df_devices[df_devices['device_level']=='npc_code_match_CLN_Device_Serial_Number']))
    logger.info("Device code level matching completed for serial number column. Number of records labelled at MPC code level: {}", len(df_devices[df_devices['device_level']=='mpc_code_match_CLN_Device_Serial_Number']))
    logger.info("Device code level matching completed for serial number column. Number of records labelled at EAN code level: {}", len(df_devices[df_devices['device_level']=='ean_code_match_CLN_Device_Serial_Number']))

    df_devices = device_code_level_matching(df_devices, df_catalogue, 'matched_device', 'CLN_Manufacturer_Device_Name', logger)
    logger.info("Device code level matching completed for manufacturer name column. Number of records labelled at NPC code level: {}", len(df_devices[df_devices['device_level']=='npc_code_match_CLN_Manufacturer_Device_Name']))
    logger.info("Device code level matching completed for manufacturer name column. Number of records labelled at MPC code level: {}", len(df_devices[df_devices['device_level']=='mpc_code_match_CLN_Manufacturer_Device_Name']))
    logger.info("Device code level matching completed for manufacturer name column. Number of records labelled at EAN code level: {}", len(df_devices[df_devices['device_level']=='ean_code_match_CLN_Manufacturer_Device_Name']))

    df_devices = clean_data(df_devices, 'CLN_Manufacturer_Device_Name', 'device_name_tokens', device_tokens_to_remove, split_numbers=False)
    df_catalogue = clean_data(df_catalogue, 'Base Description', 'primary_description_tokens', device_tokens_to_remove, split_numbers=False)
    df_devices = exact_match_with_supplier_filter(df_catalogue, df_devices, 'primary_description_tokens', 'NPC', 'device_name_tokens', 'matched_device', 'Supplier', 'Supplier')
    logger.info("Exact match level for base description + supplier matching completed. Number of records labelled: {}", len(df_devices[df_devices['device_level']=='exact_match_supplier_and_primary_description_tokens']))
    df_catalogue = clean_data(df_catalogue, 'Secondary Description', 'secondary_description_tokens', device_tokens_to_remove, split_numbers=False)
    df_devices = exact_match_with_supplier_filter(df_catalogue, df_devices, 'secondary_description_tokens', 'NPC', 'device_name_tokens', 'matched_device', 'Supplier', 'Supplier')
    logger.info("Exact match level for secondary description + supplier matching completed. Number of records labelled: {}", len(df_devices[df_devices['device_level']=='exact_match_supplier_and_secondary_description_tokens']))

    df_devices = bag_of_words_matching(df_catalogue, df_devices, 'catalogue_device_name_and_manufacturer_tokens', 'supplier_and_device_name_tokens', score_threshold=0.5)
    logger.info("Bag of words level matching completed. Number of records labelled at bag of words level: {}", len(df_devices[df_devices['device_level']=='bag_of_words_match_0.5']))

    df_devices = bag_of_words_matching(df_catalogue, df_devices, 'catalogue_device_only_tokens', 'device_name_tokens', score_threshold=0.3)
    logger.info("Bag of words with just device name tokens level matching completed. Number of records labelled at bag of words level: {}", len(df_devices[df_devices['device_level']=='bag_of_words_match_0.3']))

    logger.info("Percentage of rows that remain unmatched for device: {:,.2f}%", (len(df_devices[df_devices['matched_device'].isnull()])/len(df_devices))*100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'data/outputs/matching_results_devices{timestamp}.csv'
    df_devices.to_csv(output_path, index=False)
    logger.info("Unmatched rows saved to {}", output_path)

    logger.info("Matching Devices pipeline completed successfully.\n")
