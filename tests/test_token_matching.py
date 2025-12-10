import pandas as pd
import pytest
from unittest.mock import Mock
from utils.matching_utils import number_of_tokens_match


class TestNumberOfTokensMatch:
    """Tests for the utils.number_of_tokens_match function"""

    def test_filters_unlabeled_rows_only(self):
        """Test that only unlabeled rows are processed."""
        mock_logger = Mock()
        
        reference_df = pd.DataFrame({
            'catalogue_id': [1, 2, 3, 4],
            'Supplier_tokens': ['johnson johnson', 'abbott labs', 'medtronic inc', 'gore'],
            'Brand': ['JJ Brand', 'Abbott Brand', 'Medtronic Brand', 'Gore']
        })

        to_match_df = pd.DataFrame({
            'index': [10, 11, 12, 13],
            'CLN_manufacturer_tokens_list': [
                ['johnson', 'johnson'],
                ['abbott', 'labs'],
                ['gore'],
                ['medtronic']
            ],
            'Manufacturer_label': [None, None, 'pre_matched', None],
            'device_name': ['Device A', 'Device B', 'Device C', 'Device D']
        })

        result = number_of_tokens_match(
            mock_logger,
            to_match_df,
            reference_df,
            label_col_name='Manufacturer_label',
            score_col_name='score',
            col_to_match='CLN_manufacturer_tokens_list',
            reference_col='Supplier_tokens',
            reference_labels_col='catalogue_id',
            score_threshold=0.5
        )
        
        expected = pd.DataFrame({
            'index': [10, 11, 12, 13],
            'CLN_manufacturer_tokens_list': [
                ['johnson', 'johnson'],
                ['abbott', 'labs'],
                ['gore'],
                ['medtronic']
            ],
            'Manufacturer_label': [1, 2, 'pre_matched', 3],
            'device_name': ['Device A', 'Device B', 'Device C', 'Device D'],
            'level': ['token_overlap_match_CLN_manufacturer_tokens_list', 
                      'token_overlap_match_CLN_manufacturer_tokens_list', 
                      None,
                      'token_overlap_match_CLN_manufacturer_tokens_list']
        })
        print(expected.head())
        print(result.head())
        
        pd.testing.assert_frame_equal(result, expected, check_like=False)