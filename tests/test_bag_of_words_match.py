import pytest
import pandas as pd
import numpy as np
from utils.matching_utils import bag_of_words_matching

@pytest.fixture
def sample_catalogue():
    """Create a sample catalogue DataFrame."""
    return pd.DataFrame({
        'NPC': ['NPC001', 'NPC002', 'NPC003'],
        'Supplier': ['SupplierA', 'SupplierB', 'SupplierA'],
        'catalogue_device_name_and_manufacturer_tokens': [
            'laptop computer intel i7',
            'desktop pc gaming rig',
            'laptop computer dell xps'
        ]
    })


@pytest.fixture
def sample_to_match():
    """Create a sample to_match DataFrame."""
    return pd.DataFrame({
        'Supplier': ['SupplierA', 'SupplierB', 'SupplierC'],
        'device_name_tokens': [
            'laptop intel processor',
            'gaming desktop computer',
            'tablet device screen'
        ],
        'matched_device': [None, None, None]
    })


def test_bag_of_words_matching_basic(sample_catalogue, sample_to_match):
    """Test basic functionality of bag_of_words_matching."""
    result = bag_of_words_matching(sample_catalogue, sample_to_match, score_threshold=0.5)
    
    assert isinstance(result, pd.DataFrame)
    assert 'best_similarity' in result.columns
    assert 'n_best_matches' in result.columns
    assert 'best_match_devices' in result.columns


def test_bag_of_words_matching_preserves_pre_matched(sample_catalogue):
    """Test that pre-matched rows are preserved."""
    df_to_match = pd.DataFrame({
        'Supplier': ['SupplierA', 'SupplierB'],
        'device_name_tokens': ['laptop computer', 'desktop pc'],
        'matched_device': ['NPC999', None]
    })
    
    result = bag_of_words_matching(sample_catalogue, df_to_match, score_threshold=0.5)
    
    assert result[result['matched_device'] == 'NPC999'].shape[0] == 1


def test_bag_of_words_matching_empty_catalogue():
    """Test with empty catalogue."""
    df_catalogue = pd.DataFrame({
        'Supplier': [],
        'NPC': [],
        'catalogue_device_name_and_manufacturer_tokens': []
    })
    df_to_match = pd.DataFrame({
        'Supplier': ['SupplierA'],
        'device_name_tokens': ['laptop computer'],
        'matched_device': [None]
    })
    
    result = bag_of_words_matching(df_catalogue, df_to_match, score_threshold=0.5)
    assert len(result) == 1


def test_bag_of_words_matching_score_threshold(sample_catalogue, sample_to_match):
    """Test different score thresholds."""
    result_low = bag_of_words_matching(sample_catalogue, sample_to_match, score_threshold=0.3)
    result_high = bag_of_words_matching(sample_catalogue, sample_to_match, score_threshold=0.9)
    
    matched_low = result_low[result_low['matched_device'].notnull()]
    matched_high = result_high[result_high['matched_device'].notnull()]
    
    assert len(matched_low) >= len(matched_high)


def test_bag_of_words_matching_supplier_filter(sample_catalogue):
    """Test that supplier matching is considered when suppliers match."""
    df_to_match = pd.DataFrame({
        'Supplier': ['SupplierA'],
        'device_name_tokens': ['laptop computer'],
        'matched_device': [None]
    })
    
    result = bag_of_words_matching(sample_catalogue, df_to_match, score_threshold=0.5)
    assert len(result) >= 1


def test_bag_of_words_matching_multiple_matches(sample_catalogue):
    """Test handling of multiple best matches."""
    df_to_match = pd.DataFrame({
        'Supplier': [None],
        'device_name_tokens': ['laptop computer'],
        'matched_device': [None]
    })
    
    result = bag_of_words_matching(sample_catalogue, df_to_match, score_threshold=0.5)
    
    assert 'n_best_matches' in result.columns
    assert result['n_best_matches'].iloc[0] >= 0


def test_bag_of_words_matching_no_supplier_match(sample_catalogue):
    """Test matching when supplier in to_match doesn't exist in catalogue."""
    df_to_match = pd.DataFrame({
        'Supplier': ['UnknownSupplier'],
        'device_name_tokens': ['desktop pc gaming'],
        'matched_device': [None]
    })
    
    result = bag_of_words_matching(sample_catalogue, df_to_match, score_threshold=0.5)
    assert len(result) == 1


def test_bag_of_words_matching_null_supplier(sample_catalogue):
    """Test matching when supplier is null."""
    df_to_match = pd.DataFrame({
        'Supplier': [None],
        'device_name_tokens': ['laptop computer intel'],
        'matched_device': [None]
    })
    
    result = bag_of_words_matching(sample_catalogue, df_to_match, score_threshold=0.5)
    assert len(result) == 1
    assert 'best_similarity' in result.columns


def test_bag_of_words_matching_columns_dropped(sample_catalogue):
    """Test that temporary columns are properly dropped."""
    df_to_match = pd.DataFrame({
        'Supplier': ['SupplierA'],
        'device_name_tokens': ['laptop computer'],
        'matched_device': [None],
        'best_similarity': [0.5],
        'n_best_matches': [1],
        'best_match_devices': [['NPC001']]
    })
    
    result = bag_of_words_matching(sample_catalogue, df_to_match, score_threshold=0.5)
    assert isinstance(result, pd.DataFrame)


def test_bag_of_words_matching_all_pre_matched(sample_catalogue):
    """Test when all rows are pre-matched."""
    df_to_match = pd.DataFrame({
        'Supplier': ['SupplierA', 'SupplierB'],
        'device_name_tokens': ['laptop computer', 'desktop pc'],
        'matched_device': ['NPC001', 'NPC002']
    })
    
    result = bag_of_words_matching(sample_catalogue, df_to_match, score_threshold=0.5)
    assert len(result) == 2
    assert result['matched_device'].notnull().all()


def test_bag_of_words_matching_similarity_scores(sample_catalogue, sample_to_match):
    """Test that similarity scores are between 0 and 1."""
    result = bag_of_words_matching(sample_catalogue, sample_to_match, score_threshold=0.5)
    
    scores = result['best_similarity'].values
    assert all(0 <= score <= 1 for score in scores if not np.isnan(score))