import pandas as pd
from loguru import logger
import pytest

from utils.matching_utils import jaro_winkler_match

class TestExactMatching:
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
                'Manufacturer_label': pd.Series([pd.NA], dtype="Int64"),
                'level': [None]
            }
        )

        output_df = jaro_winkler_match(logger, df_devices, reference_df, 'Manufacturer_label', 'Supplier_score', 'CLN_Manufacturer_tokens', 'Supplier_tokens','catalogue_manufacturers_index', 0.86)

        pd.testing.assert_frame_equal(output_df, expected)

    def test_mix_of_match_and_no_match(self):
        reference_df = pd.DataFrame(
            {
                'catalogue_manufacturers_index': [0, 1, 2, 3],
                'Supplier_tokens': ['abbott', 'covidien', 'smith nephew', 'polar'],
            }
        )

        df_devices = pd.DataFrame(
            {
                'index': [0, 1, 2, 3],
                'CLN_Manufacturer_tokens': ['smith nephew', 'covidien', 'polaris', 'manufacturer'],
                'Manufacturer_label': [2, 1, None, None],
                'level': ['exact_match_Supplier_tokens', 'exact_match_Supplier_tokens', None, None]

            }
        )

        expected = pd.DataFrame(
            {
                'index': [0, 1, 2, 3],
                'CLN_Manufacturer_tokens': ['smith nephew', 'covidien', 'polaris', 'manufacturer'],
                'Manufacturer_label': pd.Series([2, 1, 3, None], dtype="Int64"),
                'level': ['exact_match_Supplier_tokens', 'exact_match_Supplier_tokens', 'jaro_winkler_match_Supplier_tokens', None]
            }
        )

        output_df = jaro_winkler_match(logger, df_devices, reference_df, 'Manufacturer_label', 'Supplier_score', 'CLN_Manufacturer_tokens', 'Supplier_tokens','catalogue_manufacturers_index', 0.86)

        pd.testing.assert_frame_equal(output_df, expected)

    def test_competing_good_scores(self):
        reference_df = pd.DataFrame(
            {
                'catalogue_manufacturers_index': [0, 1, 2, 3],
                'Supplier_tokens': ['abbott', 'polaris', 'smith nephew', 'polar'],
            }
        )

        df_devices = pd.DataFrame(
            {
                'index': [0, 1, 2, 3],
                'CLN_Manufacturer_tokens': ['smith nephew', 'covidien', 'polaris', 'manufacturer'],
                'Manufacturer_label': [2, 1, None, None],
                'level': ['exact_match_Supplier_tokens', 'exact_match_Supplier_tokens', None, None]

            }
        )

        expected = pd.DataFrame(
            {
                'index': [0, 1, 2, 3],
                'CLN_Manufacturer_tokens': ['smith nephew', 'covidien', 'polaris', 'manufacturer'],
                'Manufacturer_label': pd.Series([2, 1, 1, None], dtype="Int64"),
                'level': ['exact_match_Supplier_tokens', 'exact_match_Supplier_tokens', 'jaro_winkler_match_Supplier_tokens', None]
            }
        )

        output_df = jaro_winkler_match(logger, df_devices, reference_df, 'Manufacturer_label', 'Supplier_score', 'CLN_Manufacturer_tokens', 'Supplier_tokens','catalogue_manufacturers_index', 0.86)

        pd.testing.assert_frame_equal(output_df, expected)
