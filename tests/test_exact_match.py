import pandas as pd
import pytest

from utils.matching_utils import exact_match

class TestJaroWinklerMatching:
    """Tests for the utils.exact_match function"""

    def test_no_matches(self):
        reference_df = pd.DataFrame(
            {
                'catalogue_manufacturers_index': [0],
                'Supplier_tokens': ['abbott'],
            }
        )

        df_devices = pd.DataFrame(
            {
                'index': [0],
                'CLN_Manufacturer': ['SMITH & NEPHEW'],
                'CLN_Manufacturer_tokens': ['smith nephew'],
                'Manufacturer_label': [pd.NA]
            }
        )

        expected = pd.DataFrame(
            {
                'index': [0],
                'CLN_Manufacturer': ['SMITH & NEPHEW'],
                'CLN_Manufacturer_tokens': ['smith nephew'],
                'Manufacturer_label': [pd.NA],
                'level': [None]
            }
        )

        output_df = exact_match(reference_df, df_devices, 'Supplier_tokens', 'catalogue_manufacturers_index', 'CLN_Manufacturer_tokens', 'Manufacturer_label')

        pd.testing.assert_frame_equal(output_df, expected)

    def test_mix_of_match_and_no_match(self):
        reference_df = pd.DataFrame(
            {
                'catalogue_manufacturers_index': [0, 1, 2],
                'Supplier_tokens': ['abbott', 'covidien', 'smith nephew'],
            }
        )

        df_devices = pd.DataFrame(
            {
                'index': [0, 1, 2],
                'CLN_Manufacturer_tokens': ['smith nephew', 'covidien', 'polaris'],
                'Manufacturer_label': [None, None, None]
            }
        )

        expected = pd.DataFrame(
            {
                'index': [0, 1, 2],
                'CLN_Manufacturer_tokens': ['smith nephew', 'covidien', 'polaris'],
                'Manufacturer_label': pd.Series([2, 1, None], dtype="Int64"),
                'level': ['exact_match_Supplier_tokens', 'exact_match_Supplier_tokens', None]
            }
        )

        output_df = exact_match(reference_df, df_devices, 'Supplier_tokens', 'catalogue_manufacturers_index', 'CLN_Manufacturer_tokens', 'Manufacturer_label')
        pd.testing.assert_frame_equal(output_df, expected)
