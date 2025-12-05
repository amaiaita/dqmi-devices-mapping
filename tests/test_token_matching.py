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

    # def test_threshold_filtering_high_threshold(self):
    #     """Test that high threshold filters out low-scoring matches."""
    #     mock_logger = Mock()
        
    #     reference_df = pd.DataFrame({
    #         'catalogue_id': [1],
    #         'Supplier_tokens': ['johnson johnson'],
    #     })

    #     to_match_df = pd.DataFrame({
    #         'index': [10, 11],
    #         'CLN_manufacturer_tokens_list': [
    #             ['johnson', 'johnson'],  # perfect match
    #             ['gore']                  # no match
    #         ],
    #         'Manufacturer_label': [None, None],
    #     })

    #     result = number_of_tokens_match(
    #         mock_logger,
    #         to_match_df,
    #         reference_df,
    #         label_col_name='Manufacturer_label',
    #         score_col_name='score',
    #         col_to_match='CLN_manufacturer_tokens_list',
    #         reference_col='Supplier_tokens',
    #         reference_labels_col='catalogue_id',
    #         score_threshold=0.8
    #     )
        
    #     expected = pd.DataFrame({
    #         'index': [10, 11],
    #         'CLN_manufacturer_tokens_list': [
    #             ['johnson', 'johnson'],
    #             ['gore']
    #         ],
    #         'Manufacturer_label': [1, None],
    #         'level': ['token_overlap_match_CLN_manufacturer_tokens_list', None]
    #     })
        
    #     pd.testing.assert_frame_equal(result, expected)

    # def test_preserves_original_device_data(self):
    #     """Test that original device data is preserved in output."""
    #     mock_logger = Mock()
        
    #     reference_df = pd.DataFrame({
    #         'catalogue_id': [1],
    #         'Supplier_tokens': ['abbott labs'],
    #         'Brand': ['Abbott Brand']
    #     })

    #     to_match_df = pd.DataFrame({
    #         'index': [10, 11],
    #         'CLN_manufacturer_tokens_list': [['abbott', 'labs'], ['gore']],
    #         'Manufacturer_label': [None, None],
    #         'device_name': ['Device A', 'Device B'],
    #         'other_col': ['value1', 'value2']
    #     })

    #     result = number_of_tokens_match(
    #         mock_logger,
    #         to_match_df,
    #         reference_df,
    #         label_col_name='Manufacturer_label',
    #         score_col_name='score',
    #         col_to_match='CLN_manufacturer_tokens_list',
    #         reference_col='Supplier_tokens',
    #         reference_labels_col='catalogue_id',
    #         score_threshold=0.5
    #     )
        
    #     expected = pd.DataFrame({
    #         'index': [10, 11],
    #         'CLN_manufacturer_tokens_list': [['abbott', 'labs'], ['gore']],
    #         'Manufacturer_label': [1, None],
    #         'device_name': ['Device A', 'Device B'],
    #         'other_col': ['value1', 'value2'],
    #         'level': ['token_overlap_match_CLN_manufacturer_tokens_list', None]
    #     })
        
    #     pd.testing.assert_frame_equal(result, expected)

    # def test_mix_of_matched_and_unmatched(self):
    #     """Test handling of mix of matched and unmatched rows."""
    #     mock_logger = Mock()
        
    #     reference_df = pd.DataFrame({
    #         'catalogue_id': [1, 2],
    #         'Supplier_tokens': ['johnson johnson', 'abbott labs'],
    #     })

    #     to_match_df = pd.DataFrame({
    #         'index': [10, 11, 12],
    #         'CLN_manufacturer_tokens_list': [
    #             ['johnson', 'johnson'],  # match
    #             ['abbott', 'labs'],      # match
    #             ['polaris']              # no match
    #         ],
    #         'Manufacturer_label': [None, None, None],
    #     })

    #     result = number_of_tokens_match(
    #         mock_logger,
    #         to_match_df,
    #         reference_df,
    #         label_col_name='Manufacturer_label',
    #         score_col_name='score',
    #         col_to_match='CLN_manufacturer_tokens_list',
    #         reference_col='Supplier_tokens',
    #         reference_labels_col='catalogue_id',
    #         score_threshold=0.5
    #     )
        
    #     expected = pd.DataFrame({
    #         'index': [10, 11, 12],
    #         'CLN_manufacturer_tokens_list': [
    #             ['johnson', 'johnson'],
    #             ['abbott', 'labs'],
    #             ['polaris']
    #         ],
    #         'Manufacturer_label': [1, 2, None],
    #         'level': ['token_overlap_match_CLN_manufacturer_tokens_list', 
    #                   'token_overlap_match_CLN_manufacturer_tokens_list', 
    #                   None]
    #     })
        
    #     pd.testing.assert_frame_equal(result, expected)

    # def test_preserves_prelabeled_rows(self):
    #     """Test that pre-labeled rows are not re-processed."""
    #     mock_logger = Mock()
        
    #     reference_df = pd.DataFrame({
    #         'catalogue_id': [1, 2],
    #         'Supplier_tokens': ['johnson johnson', 'abbott labs'],
    #     })

    #     to_match_df = pd.DataFrame({
    #         'index': [10, 11, 12],
    #         'CLN_manufacturer_tokens_list': [
    #             ['johnson', 'johnson'],
    #             ['abbott', 'labs'],
    #             ['gore']
    #         ],
    #         'Manufacturer_label': [None, 999, None],  # 999 is pre-labeled
    #     })

    #     result = number_of_tokens_match(
    #         mock_logger,
    #         to_match_df,
    #         reference_df,
    #         label_col_name='Manufacturer_label',
    #         score_col_name='score',
    #         col_to_match='CLN_manufacturer_tokens_list',
    #         reference_col='Supplier_tokens',
    #         reference_labels_col='catalogue_id',
    #         score_threshold=0.5
    #     )
        
    #     expected = pd.DataFrame({
    #         'index': [10, 11, 12],
    #         'CLN_manufacturer_tokens_list': [
    #             ['johnson', 'johnson'],
    #             ['abbott', 'labs'],
    #             ['gore']
    #         ],
    #         'Manufacturer_label': [1, 999, None],
    #         'level': ['token_overlap_match_CLN_manufacturer_tokens_list', None, None]
    #     })
        
    #     pd.testing.assert_frame_equal(result, expected)