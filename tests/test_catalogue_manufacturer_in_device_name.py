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
    
    def test_basic_matching(self, sample_catalogue):
        """Test basic manufacturer matching in device name."""
        devices = pd.DataFrame({
            'device_name': [
                'Philips Monitor X200',
                'Siemens ACUSON Ultrasound',
                'Unknown Brand Device',
            ],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        expected = pd.DataFrame({
            'device_name': [
                'Philips Monitor X200',
                'Siemens ACUSON Ultrasound',
                'Unknown Brand Device',
            ],
            'Manufacturer_label': [1, 2, np.nan],
            'level': [
                'catalogue_manufacturer_in_device_name_match_Supplier',
                'catalogue_manufacturer_in_device_name_match_Supplier',
                np.nan
            ]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
    def test_case_insensitive_matching(self, sample_catalogue):
        """Test that matching is case-insensitive."""
        devices = pd.DataFrame({
            'device_name': ['PHILIPS monitor', 'siemens device', 'GE HEALTHCARE scanner'],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        expected = pd.DataFrame({
            'device_name': ['PHILIPS monitor', 'siemens device', 'GE HEALTHCARE scanner'],
            'Manufacturer_label': [1, 2, 3],
            'level': [
                'catalogue_manufacturer_in_device_name_match_Supplier',
                'catalogue_manufacturer_in_device_name_match_Supplier',
                'catalogue_manufacturer_in_device_name_match_Supplier'
            ]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
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
        
        expected = pd.DataFrame({
            'device_name': ['Philips Healthcare Monitor'],
            'Manufacturer_label': [2],
            'level': ['catalogue_manufacturer_in_device_name_match_Supplier']
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
    def test_null_device_names(self, sample_catalogue):
        """Test handling of null/NaN device names."""
        devices = pd.DataFrame({
            'device_name': [np.nan, None, ''],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        expected = pd.DataFrame({
            'device_name': [np.nan, None, ''],
            'Manufacturer_label': [np.nan, np.nan, np.nan],
            'level': [np.nan, np.nan, np.nan]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
    def test_preserve_prelabelled_rows(self, sample_catalogue):
        """Test that already-labelled rows are preserved."""
        devices = pd.DataFrame({
            'device_name': ['Philips Monitor', 'Unknown Device'],
            'Manufacturer_label': [99, None],
            'level': ['manual_label', None]
        })
        
        expected = pd.DataFrame({
            'device_name': ['Philips Monitor', 'Unknown Device'],
            'Manufacturer_label': [99, np.nan],
            'level': ['manual_label', np.nan]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        # Sort by device_name to handle potential row reordering by the function
        pd.testing.assert_frame_equal(
            result.sort_values('device_name').reset_index(drop=True), 
            expected.sort_values('device_name').reset_index(drop=True), 
            check_dtype=False
        )
    
    def test_word_boundary_matching(self, sample_catalogue):
        """Test that partial word matches are NOT matched (requires word boundaries)."""
        devices = pd.DataFrame({
            'device_name': [
                'Fiber coil device',
                'Siemens product',
                'Philipslike device'
            ],
            'Manufacturer_label': [None, None, None],
            'level': [None, None, None]
        })
        
        expected = pd.DataFrame({
            'device_name': [
                'Fiber coil device',
                'Siemens product',
                'Philipslike device'
            ],
            'Manufacturer_label': [np.nan, 2, np.nan],
            'level': [np.nan, 'catalogue_manufacturer_in_device_name_match_Supplier', np.nan]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            sample_catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
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
        
        expected = pd.DataFrame({
            'device_name': [
                'Philips Monitor',
                'Device with spaces',
                'Siemens equipment',
                'GE scanner'
            ],
            'Manufacturer_label': [1, np.nan, 4, 6],
            'level': [
                'catalogue_manufacturer_in_device_name_match_Supplier',
                np.nan,
                'catalogue_manufacturer_in_device_name_match_Supplier',
                'catalogue_manufacturer_in_device_name_match_Supplier'
            ]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
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
        
        expected = pd.DataFrame({
            'device_name': [
                'Philips Monitor X200',
                'Siemens ACUSON Ultrasound',
                'GE Healthcare ECG'
            ],
            'Manufacturer_label': [1, 2, 3],
            'level': [
                'catalogue_manufacturer_in_device_name_match_Brand',
                'catalogue_manufacturer_in_device_name_match_Brand',
                'catalogue_manufacturer_in_device_name_match_Brand'
            ]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Brand',
            logger
        )
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
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
        
        expected = pd.DataFrame({
            'device_name': ['Philips Monitor', 'None Device'],
            'Manufacturer_label': [1, np.nan],
            'level': [
                'catalogue_manufacturer_in_device_name_match_Supplier',
                np.nan
            ]
        })
        
        result = catalogue_manufacturer_in_device_name_matching(
            devices,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
    def test_row_count_preserved(self, sample_catalogue):
        """Test that the result has the same number of rows."""
        devices = pd.DataFrame({
            'device_name': ['Philips Monitor', 'Unknown Device', 'Siemens System'],
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
        
        assert len(result) == len(devices)
    
    def test_all_columns_preserved(self, sample_catalogue):
        """Test that all columns from input are preserved in output."""
        devices = pd.DataFrame({
            'device_name': ['Philips Monitor'],
            'Manufacturer_label': [None],
            'level': [None],
            'extra_col': ['value1']
        })
        
        expected = pd.DataFrame({
            'device_name': ['Philips Monitor'],
            'Manufacturer_label': [1],
            'level': ['catalogue_manufacturer_in_device_name_match_Supplier'],
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
        
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
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
        assert result.loc[0, 'level'] == 'catalogue_manufacturer_in_device_name_match_Supplier'
    
    def test_supplier_vs_brand_matching(self):
        """Test that brand and supplier columns can produce different results."""
        catalogue = pd.DataFrame({
            'Supplier': ['Philips Electronics', 'Siemens Healthcare'],
            'Brand': ['Philips', 'ACUSON'],
            'catalogue_manufacturers_index': [1, 2]
        })
        
        devices_supplier = pd.DataFrame({
            'device_name': [
                'Device with Philips Electronics name',
                'Product with ACUSON inside'
            ],
            'Manufacturer_label': [None, None],
            'level': [None, None]
        })
        
        devices_brand = pd.DataFrame({
            'device_name': [
                'Device with Philips Electronics name',
                'Product with ACUSON inside'
            ],
            'Manufacturer_label': [None, None],
            'level': [None, None]
        })
        
        # Test with Supplier column
        result_supplier = catalogue_manufacturer_in_device_name_matching(
            devices_supplier,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Supplier',
            logger
        )
        
        expected_supplier = pd.DataFrame({
            'device_name': [
                'Device with Philips Electronics name',
                'Product with ACUSON inside'
            ],
            'Manufacturer_label': [1, np.nan],
            'level': [
                'catalogue_manufacturer_in_device_name_match_Supplier',
                np.nan
            ]
        })
        
        pd.testing.assert_frame_equal(result_supplier, expected_supplier, check_dtype=False)
        
        # Test with Brand column
        result_brand = catalogue_manufacturer_in_device_name_matching(
            devices_brand,
            catalogue,
            'Manufacturer_label',
            'device_name',
            'Brand',
            logger
        )
        
        expected_brand = pd.DataFrame({
            'device_name': [
                'Device with Philips Electronics name',
                'Product with ACUSON inside'
            ],
            'Manufacturer_label': [1, 2],
            'level': [
                'catalogue_manufacturer_in_device_name_match_Brand',
                'catalogue_manufacturer_in_device_name_match_Brand'
            ]
        })
        
        pd.testing.assert_frame_equal(result_brand, expected_brand, check_dtype=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
