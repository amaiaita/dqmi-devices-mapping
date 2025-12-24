import pandas as pd
import numpy as np
from utils.matching_utils import exact_match_with_supplier_filter

def test_exact_match_with_supplier_filter_assigns_label_correctly():
    reference_df = pd.DataFrame({
        "device_name": ["Aspirin Pump", "Insulin Pump"],
        "label": [1, 2],
        "supplier": ["SupplierA", "SupplierB"]
    })

    df2 = pd.DataFrame({
        "raw_device_name": ["Aspirin Pump", "Insulin Pump", "Aspirin Pump"],
        "supplier_name": ["SupplierA", "SupplierB", "SupplierB"],
        "device_label": [np.nan, np.nan, np.nan],
        "NPC": [1, 2, 3]
    })

    result = exact_match_with_supplier_filter(
        reference_df=reference_df,
        df2=df2,
        col_df_1_for_comparison="device_name",
        col_df_1_for_labels="label",
        col_df_2="raw_device_name",
        col_df2_label="device_label",
        df_2_supplier_col="supplier_name",
        df_reference_supplier_col="supplier",
    )

    matched = result[
        (result["raw_device_name"] == "Aspirin Pump") &
        (result["supplier_name"] == "SupplierA")
    ].iloc[0]

    assert matched["device_label"] == 1
    assert matched["device_level"] == "exact_match_supplier_and_device_name"
