import pytest
import pandas as pd
import numpy as np
from utils.matching_utils import substring_match, find_substring_match


class TestFindSubstringMatch:
    """Test cases for the find_substring_match helper function."""
    
    def test_reference_substring_in_text(self):
        """Test when reference text is a substring of the text."""
        text = "gaming laptop pro"
        reference_texts = ["laptop", "monitor", "desktop"]
        reference_dict = {
            "laptop": "REF001",
            "monitor": "REF002",
            "desktop": "REF003"
        }
        result = find_substring_match(text, reference_texts, reference_dict)
        assert result == "REF001"
    
    def test_case_insensitive_match(self):
        """Test that matching is case insensitive."""
        text = "GAMING LAPTOP PRO"
        reference_texts = ["laptop", "monitor"]
        reference_dict = {
            "laptop": "REF001",
            "monitor": "REF002"
        }
        result = find_substring_match(text, reference_texts, reference_dict)
        assert result == "REF001"
    
    def test_no_reference_substring_in_text(self):
        """Test when reference text is not a substring of text."""
        text = "gaming device"
        reference_texts = ["laptop", "monitor", "printer"]
        reference_dict = {
            "laptop": "REF001",
            "monitor": "REF002",
            "printer": "REF003"
        }
        result = find_substring_match(text, reference_texts, reference_dict)
        assert result is None
    
    def test_empty_text(self):
        """Test with empty text."""
        text = ""
        reference_texts = ["laptop", "monitor"]
        reference_dict = {
            "laptop": "REF001",
            "monitor": "REF002"
        }
        result = find_substring_match(text, reference_texts, reference_dict)
        assert result is None
    
    def test_empty_reference_texts(self):
        """Test with empty reference texts list."""
        text = "laptop"
        reference_texts = []
        reference_dict = {}
        result = find_substring_match(text, reference_texts, reference_dict)
        assert result is None
    
    def test_multiple_matches_returns_first(self):
        """Test that when multiple references match, the first one is returned."""
        text = "laptop monitor gaming"
        reference_texts = ["laptop", "monitor", "gaming"]
        reference_dict = {
            "laptop": "REF001",
            "monitor": "REF002",
            "gaming": "REF003"
        }
        result = find_substring_match(text, reference_texts, reference_dict)
        # Should return first match found
        assert result == "REF001"
    
    def test_numeric_text(self):
        """Test with numeric text."""
        text = "model 123 device"
        reference_texts = ["123", "456"]
        reference_dict = {
            "123": "NUM001",
            "456": "NUM002"
        }
        result = find_substring_match(text, reference_texts, reference_dict)
        assert result == "NUM001"


class TestSubstringMatch:
    """Test cases for the substring_match function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframes for testing."""
        df_to_match = pd.DataFrame({
            'device_name': ['gaming laptop', 'office printer', 'monitor curved', 'keyboard wireless'],
            'Manufacturer_label': [None, None, None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'printer', 'monitor', 'keyboard'],
            'reference_label': ['LAPTOP001', 'PRINTER001', 'MONITOR001', 'KEYBOARD001']
        })
        
        return df_to_match, df_reference
    
    def test_basic_substring_matching(self, sample_data):
        """Test basic substring matching functionality."""
        df_to_match, df_reference = sample_data
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # All rows should have matches
        assert result['Manufacturer_label'].notna().sum() == 4
        assert result.loc[0, 'Manufacturer_label'] == 'LAPTOP001'
        assert result.loc[1, 'Manufacturer_label'] == 'PRINTER001'
        assert result.loc[2, 'Manufacturer_label'] == 'MONITOR001'
        assert result.loc[3, 'Manufacturer_label'] == 'KEYBOARD001'
    
    def test_preserve_pre_matched_rows(self):
        """Test that pre-matched rows are preserved."""
        df_to_match = pd.DataFrame({
            'device_name': ['gaming laptop', 'office printer', 'unknown device'],
            'Manufacturer_label': ['EXISTING_001', None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'printer'],
            'reference_label': ['LAPTOP001', 'PRINTER001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # First row should retain its pre-matched label
        assert result.loc[0, 'Manufacturer_label'] == 'EXISTING_001'
        # Other rows should have new matches
        assert result.loc[1, 'Manufacturer_label'] == 'PRINTER001'
        assert result.loc[2, 'Manufacturer_label'] is None or pd.isna(result.loc[2, 'Manufacturer_label'])
    
    def test_no_matches_found(self):
        """Test behavior when no substring matches are found."""
        df_to_match = pd.DataFrame({
            'device_name': ['xyz123', 'abc456'],
            'Manufacturer_label': [None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'monitor'],
            'reference_label': ['REF001', 'REF002']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # No matches should be found
        assert result['Manufacturer_label'].isna().sum() == 2
    
    def test_match_level_column_set_for_matches(self):
        """Test that match_level column is set correctly."""
        df_to_match = pd.DataFrame({
            'device_name': ['gaming laptop', 'phone device'],
            'Manufacturer_label': [None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop'],
            'reference_label': ['LAPTOP001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Matched row should have level set
        assert result.loc[0, 'match_level'] == 'substring_match_reference_name'
        # Non-matched row should not have level set
        assert pd.isna(result.loc[1, 'match_level'])
    
    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        df_to_match = pd.DataFrame({
            'device_name': ['GAMING LAPTOP', 'OFFICE PRINTER'],
            'Manufacturer_label': [None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'printer'],
            'reference_label': ['LAPTOP001', 'PRINTER001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Both should match despite different casing
        assert result.loc[0, 'Manufacturer_label'] == 'LAPTOP001'
        assert result.loc[1, 'Manufacturer_label'] == 'PRINTER001'
    
    def test_preserves_dataframe_structure(self):
        """Test that the result dataframe has expected structure."""
        df_to_match = pd.DataFrame({
            'device_name': ['gaming laptop', 'office printer'],
            'Manufacturer_label': [None, None],
            'other_col': ['A', 'B']
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop'],
            'reference_label': ['LAPTOP001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Result should contain all original columns from df_to_match
        assert 'device_name' in result.columns
        assert 'Manufacturer_label' in result.columns
        assert 'other_col' in result.columns
        assert 'match_level' in result.columns
        # Should have 2 rows
        assert len(result) == 2
    
    def test_handles_null_values_in_data(self):
        """Test handling of null/NaN values."""
        df_to_match = pd.DataFrame({
            'device_name': ['gaming laptop', np.nan, 'curved monitor'],
            'Manufacturer_label': [None, None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'monitor'],
            'reference_label': ['LAPTOP001', 'MONITOR001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Rows with actual data should match
        assert result.loc[0, 'Manufacturer_label'] == 'LAPTOP001'
        assert result.loc[2, 'Manufacturer_label'] == 'MONITOR001'
        # Result should have 3 rows
        assert len(result) == 3
    
    def test_multiple_pre_matched_rows(self):
        """Test with multiple pre-matched rows."""
        df_to_match = pd.DataFrame({
            'device_name': ['gaming laptop', 'office printer', 'curved monitor', 'wireless keyboard'],
            'Manufacturer_label': ['EXISTING_001', None, 'EXISTING_002', None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'printer', 'keyboard'],
            'reference_label': ['LAPTOP001', 'PRINTER001', 'KEYBOARD001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Pre-matched rows should be preserved
        assert result.loc[0, 'Manufacturer_label'] == 'EXISTING_001'
        assert result.loc[2, 'Manufacturer_label'] == 'EXISTING_002'
        # Unmatched pre-labeled rows should keep their labels
        assert result.loc[1, 'Manufacturer_label'] == 'PRINTER001'
        assert result.loc[3, 'Manufacturer_label'] == 'KEYBOARD001'
    
    def test_partial_text_match(self):
        """Test that partial text is matched correctly."""
        df_to_match = pd.DataFrame({
            'device_name': ['Dell laptop XPS', 'HP printer LaserJet'],
            'Manufacturer_label': [None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'printer'],
            'reference_label': ['LAPTOP001', 'PRINTER001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Both should match
        assert result.loc[0, 'Manufacturer_label'] == 'LAPTOP001'
        assert result.loc[1, 'Manufacturer_label'] == 'PRINTER001'


class TestSubstringMatchEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframes(self):
        """Test with empty to_match dataframe."""
        df_to_match = pd.DataFrame({
            'device_name': [],
            'Manufacturer_label': []
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop'],
            'reference_label': ['LAPTOP001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Should return empty dataframe without errors
        assert len(result) == 0
    
    def test_empty_reference_dataframe(self):
        """Test with empty reference dataframe."""
        df_to_match = pd.DataFrame({
            'device_name': ['gaming laptop', 'office printer'],
            'Manufacturer_label': [None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': [],
            'reference_label': []
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # No matches should be found
        assert result['Manufacturer_label'].isna().sum() == 2
    
    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        df_to_match = pd.DataFrame({
            'device_name': ['laptop-pro gaming', 'printer_3000 office', 'monitor (27in) curved'],
            'Manufacturer_label': [None, None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'printer', 'monitor'],
            'reference_label': ['LAPTOP001', 'PRINTER001', 'MONITOR001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # All should match despite special characters
        assert result['Manufacturer_label'].notna().sum() == 3
    
    def test_single_character_reference(self):
        """Test with single character reference."""
        df_to_match = pd.DataFrame({
            'device_name': ['apple', 'banana', 'zebra'],
            'Manufacturer_label': [None, None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['a', 'b', 'z'],
            'reference_label': ['REF_A', 'REF_B', 'REF_Z']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # All single characters should be found as substrings
        assert result['Manufacturer_label'].notna().sum() == 3
    
    def test_duplicate_reference_values(self):
        """Test with duplicate reference values (different labels)."""
        df_to_match = pd.DataFrame({
            'device_name': ['gaming laptop', 'office laptop'],
            'Manufacturer_label': [None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'laptop'],
            'reference_label': ['LAPTOP001', 'LAPTOP002']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Both rows should get matched (note: dict.set_index will keep last value for duplicates)
        assert result['Manufacturer_label'].notna().sum() >= 1
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in text."""
        df_to_match = pd.DataFrame({
            'device_name': ['  gaming laptop  ', 'office   printer'],
            'Manufacturer_label': [None, None]
        })
        
        df_reference = pd.DataFrame({
            'reference_name': ['laptop', 'printer'],
            'reference_label': ['LAPTOP001', 'PRINTER001']
        })
        
        result = substring_match(
            df_to_match=df_to_match.copy(),
            df_reference=df_reference,
            label_col_name='Manufacturer_label',
            col_to_match='device_name',
            reference_col='reference_name',
            reference_labels_col='reference_label',
            level_col='match_level'
        )
        
        # Should still match despite whitespace
        assert result['Manufacturer_label'].notna().sum() == 2
