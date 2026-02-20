import pytest
import pandas as pd
import numpy as np
from loguru import logger
from utils.matching_utils import catalogue_manufacturer_in_device_name_matching


class TestCatalogueManufacturerInDeviceNameMatching:
    """Test suite for catalogue_manufacturer_in_device_name_matching function."""
    
    @pytest.fixture
    def sample_catalogue(self):
        """Create a sample catalogue suppliers dataframe."""
        return pd.DataFrame({
            'Supplier': ['Philips', 'Siemens', 'GE Healthcare', 'Sony', 'Philips Healthcare'],
            'catalogue_manufacturers_index': [1, 2, 3, 4, 5]
        })
    
    @pytest.fixture
    def sample_devices(self):
        """Create a sample devices dataframe to match."""
        return pd.DataFrame({
            'device_name': [
                'Philips Monitor X200',
                'Siemens ACUSON Ultrasound',
                'Unknown Brand Device',
                'GE ECG Machine',
                'Sony Camera',
                np.nan,
                ''
            ],
            'Manufacturer_label': [None, None, None, None, None, None, None],
            'level': [None, None, None, None, None, None, None]
        })
    
    def test_basic_matching(self, sample_catalogue, sample_devices):
        """Test basic manufacturer matching in device name."""
        result = catalogue_manufacturer_in_device_name_matching(
            sample_devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Check that Philips was matched
        assert result.loc[0, 'Manufacturer_label'] == 1
        assert result.loc[0, 'level'] == 'catalogue_manufacturer_in_device_name_match_Supplier'
        
        # Check that Siemens was matched
        assert result.loc[1, 'Manufacturer_label'] == 2
    
    def test_no_match(self, sample_catalogue, sample_devices):
        """Test that unmatched devices remain unmatched."""
        result = catalogue_manufacturer_in_device_name_matching(
            sample_devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Unknown Brand Device should not match
        assert pd.isna(result.loc[2, 'Manufacturer_label'])
        assert pd.isna(result.loc[2, 'level'])
    
    def test_case_insensitive_matching(self, sample_catalogue):
        """Test that matching is case-insensitive."""
        devices = pd.DataFrame({
            'device_name': ['PHILIPS monitor', 'siemens device', 'GE HEALTHCARE scanner'],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # All should match despite case differences
        assert result.loc[0, 'Manufacturer_label'] == 1
        assert result.loc[1, 'Manufacturer_label'] == 2
        assert result.loc[2, 'Manufacturer_label'] == 3
    
    def test_longest_match_first(self):
        """Test that longer manufacturer names are matched first."""
        catalogue = pd.DataFrame({
            'Supplier': ['Philips', 'Philips Healthcare'],
            'catalogue_manufacturers_index': [1, 2]
        })
        
        devices = pd.DataFrame({
            'device_name': ['Philips Healthcare Monitor'],
            'Manufacturer_label': [None],
            'level': [None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Should match "Philips Healthcare" (longer), not just "Philips"
        assert result.loc[0, 'Manufacturer_label'] == 2
    
    def test_null_device_names(self, sample_catalogue):
        """Test handling of null/NaN device names."""
        devices = pd.DataFrame({
            'device_name': [np.nan, None, ''],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # All nulls should remain unmatched
        assert pd.isna(result.loc[0, 'Manufacturer_label'])
        assert pd.isna(result.loc[1, 'Manufacturer_label'])
    
    def test_empty_string_device_name(self, sample_catalogue):
        """Test handling of empty string device names."""
        devices = pd.DataFrame({
            'device_name': ['', '   '],
            'Manufacturer_label': [None, None],
            'level': [None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Empty strings should not match
        assert pd.isna(result.loc[0, 'Manufacturer_label'])
    
    def test_preserve_prelabelled_rows(self, sample_catalogue):
        """Test that already-labelled rows are preserved."""
        devices = pd.DataFrame({
            'device_name': ['Philips Monitor', 'Unknown Device'],
            'Manufacturer_label': [99, None],  # First row is already labelled
            'level': ['manual_label', None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # First row should keep its original label
        assert result.loc[0, 'Manufacturer_label'] == 99
        assert result.loc[0, 'level'] == 'manual_label'
        
        # Second row should remain unmatched
        assert pd.isna(result.loc[1, 'Manufacturer_label'])
    
    def test_special_characters_in_names(self, sample_catalogue):
        """Test matching with special characters."""
        devices = pd.DataFrame({
            'device_name': [
                'Philips (UK) Ltd Monitor',
                'Siemens & Co. Device',
                'GE Healthcare (Advanced)'
            ],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Should still match despite special characters
        assert result.loc[0, 'Manufacturer_label'] == 1
        assert result.loc[1, 'Manufacturer_label'] == 2
        assert result.loc[2, 'Manufacturer_label'] == 3
    
    def test_partial_substring_match(self, sample_catalogue):
        """Test that substring matches work for embedded names."""
        devices = pd.DataFrame({
            'device_name': [
                'Device with Philips inside',
                'My Siemens product is great',
                'This GE Healthcare machine'
            ],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # All should match as substrings
        assert result.loc[0, 'Manufacturer_label'] == 1
        assert result.loc[1, 'Manufacturer_label'] == 2
        assert result.loc[2, 'Manufacturer_label'] == 3
    
    def test_null_manufacturers_filtered(self):
        """Test that null manufacturers in catalogue are filtered out."""
        catalogue = pd.DataFrame({
            'Supplier': ['Philips', None, 'Siemens', np.nan],
            'catalogue_manufacturers_index': [1, 2, 3, 4]
        })
        
        devices = pd.DataFrame({
            'device_name': ['Philips Monitor', 'None Device'],
            'Manufacturer_label': [None, None],
            'level': [None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Philips should match (1)
        assert result.loc[0, 'Manufacturer_label'] == 1
        
        # "None Device" should not match anything
        assert pd.isna(result.loc[1, 'Manufacturer_label'])
    
    def test_duplicate_manufacturers_in_catalogue(self):
        """Test handling of duplicate manufacturers."""
        catalogue = pd.DataFrame({
            'Supplier': ['Philips', 'Philips', 'Siemens'],
            'catalogue_manufacturers_index': [1, 2, 3]
        })
        
        devices = pd.DataFrame({
            'device_name': ['Philips Monitor'],
            'Manufacturer_label': [None],
            'level': [None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Should match one of the Philips entries (drop_duplicates removes one)
        assert result.loc[0, 'Manufacturer_label'] in [1, 2]
    
    def test_column_name_parameter(self):
        """Test that column name parameter works correctly."""
        catalogue = pd.DataFrame({
            'Brand': ['Philips', 'Siemens'],
            'catalogue_manufacturers_index': [1, 2]
        })
        
        devices = pd.DataFrame({
            'device_name': ['Philips Monitor', 'Siemens Device'],
            'Manufacturer_label': [None, None],
            'level': [None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Brand',  # Using 'Brand' instead of 'Supplier'
            logger
        )
        
        # Should match using the Brand column
        assert result.loc[0, 'Manufacturer_label'] == 1
        assert result.loc[1, 'Manufacturer_label'] == 2
    
    def test_level_column_format(self, sample_catalogue, sample_devices):
        """Test that level column has correct format with catalogue column name."""
        result = catalogue_manufacturer_in_device_name_matching(
            sample_devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Check level format includes the catalogue column name
        matched_row = result[result['Manufacturer_label'].notna()].iloc[0]
        assert matched_row['level'] == 'catalogue_manufacturer_in_device_name_match_Supplier'
    
    def test_row_count_preserved(self, sample_catalogue, sample_devices):
        """Test that the result has the same number of rows."""
        result = catalogue_manufacturer_in_device_name_matching(
            sample_devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        assert len(result) == len(sample_devices)
    
    def test_all_columns_preserved(self, sample_catalogue):
        """Test that all columns from input are preserved in output."""
        devices = pd.DataFrame({
            'device_name': ['Philips Monitor'],
            'Manufacturer_label': [None],
            'level': [None],
            'extra_col': ['value1']
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # All original columns should be present
        assert 'device_name' in result.columns
        assert 'Manufacturer_label' in result.columns
        assert 'level' in result.columns
        assert 'extra_col' in result.columns
    
    def test_with_brand_column(self):
        """Test that function works with Brand column instead of Supplier."""
        catalogue = pd.DataFrame({
            'Brand': ['Philips', 'Siemens', 'GE Healthcare'],
            'SupplierName': ['Philips Corp', 'Siemens AG', 'General Electric'],
            'catalogue_manufacturers_index': [1, 2, 3]
        })
        
        devices = pd.DataFrame({
            'device_name': [
                'Philips Monitor X200',
                'Siemens ACUSON Ultrasound',
                'GE Healthcare ECG'
            ],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Brand',  # Using Brand column
            logger
        )
        
        # All should match using Brand column
        assert result.loc[0, 'Manufacturer_label'] == 1
        assert result.loc[1, 'Manufacturer_label'] == 2
        assert result.loc[2, 'Manufacturer_label'] == 3
        
        # Check level format includes the catalogue column name
        assert result.loc[0, 'level'] == 'catalogue_manufacturer_in_device_name_match_Brand'
    
    def test_empty_and_whitespace_manufacturers_filtered(self):
        """Test that empty strings and whitespace-only manufacturers are filtered out."""
        catalogue = pd.DataFrame({
            'Supplier': ['Philips', '', '   ', 'Siemens', '\t', 'GE'],
            'catalogue_manufacturers_index': [1, 2, 3, 4, 5, 6]
        })
        
        devices = pd.DataFrame({
            'device_name': [
                'Philips Monitor',
                'Device with spaces',
                'Siemens equipment',
                'GE scanner'
            ],
            'Manufacturer_label': [None, None, None, None],
            'level': [None, None, None, None]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Valid manufacturers should match
        assert result.loc[0, 'Manufacturer_label'] == 1  # Philips
        assert result.loc[2, 'Manufacturer_label'] == 4  # Siemens
        assert result.loc[3, 'Manufacturer_label'] == 6  # GE
        
        # Row with only spaces in catalogue shouldn't cause false matches
        assert pd.isna(result.loc[1, 'Manufacturer_label'])
    
    def test_supplier_vs_brand_matching(self):
        """Test that brand and supplier columns can produce different results."""
        catalogue = pd.DataFrame({
            'Supplier': ['Philips Electronics', 'Siemens Healthcare'],
            'Brand': ['Philips', 'ACUSON'],
            'catalogue_manufacturers_index': [1, 2]
        })
        
        devices = pd.DataFrame({
            'device_name': [
                'Device with Philips Electronics name',
                'Product with ACUSON inside'
            ],
            'Manufacturer_label': [None, None],
            'level': [None, None]
        })
        
        # Test with Supplier column
        result_supplier = catalogue_manufacturer_in_device_name_matching(
            devices.copy(),
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Test with Brand column
        result_brand = catalogue_manufacturer_in_device_name_matching(
            devices.copy(),
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Brand',
            logger
        )
        
        # Supplier matching should find "Philips Electronics"
        assert result_supplier.loc[0, 'Manufacturer_label'] == 1
        # Supplier column won't match row 1 as "Siemens Healthcare" is not in "Product with ACUSON inside"
        assert pd.isna(result_supplier.loc[1, 'Manufacturer_label'])
        
        # Brand matching - "Philips" will match row 0
        assert result_brand.loc[0, 'Manufacturer_label'] == 1
        # Brand matching - "ACUSON" will match row 1
        assert result_brand.loc[1, 'Manufacturer_label'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
