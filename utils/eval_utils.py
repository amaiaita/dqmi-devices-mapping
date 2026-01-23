# Stratified proportional sampling function
import numpy as np
import pandas as pd
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

def stratified_proportional_sample(
    df: pd.DataFrame,
    group_col: str = 'device_level',
    total_n: int = 500,
    min_per_group: int = 1,
    random_state: int = 42,
    drop_na: bool = False
):
    """Return a stratified sample of size `total_n` allocated proportionally across
    categories in `group_col`.

    Returns:
        sampled_df (pd.DataFrame), allocation (pd.Series)
    """
    df = df.copy()

    if drop_na:
        df = df[df[group_col].notna()].copy()
    else:
        df[group_col] = df[group_col].fillna('UNKNOWN')

    counts = df[group_col].value_counts()

    # If requested sample >= dataset, just shuffle and return
    if total_n >= len(df):
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True), counts

    raw = counts / counts.sum() * total_n
    n_each = np.floor(raw).astype(int)

    # Make sure each group gets at least min_per_group
    n_each = n_each.clip(lower=min_per_group)

    remaining = int(total_n - n_each.sum())
    if remaining > 0:
        frac = (raw - np.floor(raw)).sort_values(ascending=False)
        for lvl in frac.index:
            if remaining <= 0:
                break
            n_each[lvl] += 1
            remaining -= 1

    # Never allocate more than the group actually has
    n_each = n_each.clip(upper=counts)

    # Small correction if clipping changed the total
    diff = int(total_n - n_each.sum())
    if diff != 0:
        # Adjust up/down on largest groups until diff == 0
        sorted_by_size = counts.sort_values(ascending=False).index
        idx = 0
        while diff != 0:
            g = sorted_by_size[idx % len(sorted_by_size)]
            if diff > 0 and n_each[g] < counts[g]:
                n_each[g] += 1
                diff -= 1
            elif diff < 0 and n_each[g] > min_per_group:
                n_each[g] -= 1
                diff += 1
            idx += 1

    # Draw samples
    samples = []
    for lvl, n in n_each.items():
        grp = df[df[group_col] == lvl]
        n_draw = min(len(grp), int(n))
        if n_draw <= 0:
            continue
        samples.append(grp.sample(n=n_draw, random_state=random_state) if n_draw < len(grp) else grp)

    sampled = pd.concat(samples).reset_index(drop=True)
    return sampled, n_each


class ClericalReviewer:
    """Interactive reviewer optimized for faster startup on larger samples.

    Key improvements:
    - Does NOT copy the entire DataFrame on init (avoids an O(N) copy).
    - Stores annotations sparsely in a dict rather than a full DataFrame (no large allocation).
    - Exports only reviewed rows by default to avoid building a full annotated copy.
    """

    def __init__(self, df, index_col=None, random_state=42, autosave=False, filename=None, export_reviewed_only=True):
        """Create an interactive reviewer for manufacturer and device matches.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing rows to review. Must include at least:
            - 'CLN_Manufacturer'
            - 'CLN_Manufacturer_Device_Name'
            - catalogue cols: 'NPC', 'Base Description','Secondary Description','MPC','EAN/GTIN' (may be NaN)
        autosave : bool
            If True, save progress to a temp file on navigation.
        filename : str|None
            Default filename for exported CSV (timestamped if None).
        export_reviewed_only : bool
            If True, export only reviewed rows when exporting. Otherwise export full dataset with annotations.
        """
        # keep a reference to df without making a full copy (faster startup)
        self.df = df.reset_index(drop=True)
        self.N = len(self.df)
        self.idx = 0
        self.random_state = random_state
        self.autosave = autosave
        self.filename = filename or f"data/outputs/clerical_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.export_reviewed_only = export_reviewed_only

        # detect supplier/brand column names (fast)
        cols = set(self.df.columns)
        self.supplier_col = next((c for c in ['Supplier', 'Supplier_cat', 'Supplier_x', 'Supplier_y'] if c in cols), None)
        self.brand_col = next((c for c in ['Brand', 'Brand_cat'] if c in cols), None)

        # sparse results storage: {index: {col: value}}
        self.results = {}

        # build widgets once
        self._build_widgets()

        # show first row
        self._show()

    def _build_widgets(self):
        self.out = widgets.Output()

        # Text displays
        self.manuf_label = widgets.HTML()
        self.manuf_candidates = widgets.HTML()
        self.device_label = widgets.HTML()
        self.device_candidates = widgets.HTML()

        # Controls for manufacturer
        self.manuf_choice = widgets.ToggleButtons(options=['Match', 'No match', 'Unsure'], value=None)
        self.manuf_comment = widgets.Textarea(placeholder='Add comments about manufacturer match...', layout=widgets.Layout(width='100%', height='80px'))

        # Controls for device
        self.device_choice = widgets.ToggleButtons(options=['Match', 'No match', 'Unsure'], value=None)
        self.device_comment = widgets.Textarea(placeholder='Add comments about device match...', layout=widgets.Layout(width='100%', height='80px'))

        # Navigation
        self.prev_btn = widgets.Button(description='<< Previous', button_style='')
        self.next_btn = widgets.Button(description='Next >>', button_style='')
        self.jump_to = widgets.BoundedIntText(value=1, min=1, max=max(1, self.N), description='Go to:')
        self.save_btn = widgets.Button(description='Save progress', button_style='success')
        self.export_btn = widgets.Button(description='Export reviewed CSV', button_style='info')
        self.progress_label = widgets.Label()

        # bind handlers
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)
        self.jump_to.observe(self._on_jump, names='value')
        self.save_btn.on_click(self._on_save)
        self.export_btn.on_click(self._on_export)

        # layout
        manuf_box = widgets.VBox([widgets.HTML('<b>Manufacturer</b>'), self.manuf_label, self.manuf_candidates, self.manuf_choice, self.manuf_comment])
        device_box = widgets.VBox([widgets.HTML('<b>Device</b>'), self.device_label, self.device_candidates, self.device_choice, self.device_comment])
        nav_box = widgets.HBox([self.prev_btn, self.next_btn, self.jump_to, self.progress_label, self.save_btn, self.export_btn])

        self.ui = widgets.VBox([nav_box, widgets.HBox([manuf_box, device_box]), self.out])
        display(self.ui)
        self._show()

    def _render_row(self, i):
        row = self.df.iloc[i]
        manuf = row.get('CLN_Manufacturer', '')
        manuf_html = f"<pre style='white-space:pre-wrap'>{manuf}</pre>"

        brand = row.get(self.brand_col, '') if self.brand_col else ''
        supplier = row.get(self.supplier_col, '') if self.supplier_col else ''
        if brand or supplier:
            candidates_html = f"<b>Brand:</b> <pre style='white-space:pre-wrap'>{brand}</pre>\n<b>Supplier:</b> <pre style='white-space:pre-wrap'>{supplier}</pre>"
        else:
            candidates_html = "<i>No Brand/Supplier available</i>"

        device_name = row.get('CLN_Manufacturer_Device_Name', '')
        device_html = f"<pre style='white-space:pre-wrap'>{device_name}</pre>"

        npc = row.get('NPC', '')
        base = row.get('Base Description', '')
        secondary = row.get('Secondary Description', '')
        mpc = row.get('MPC', '')
        ean = row.get('EAN/GTIN', '')

        device_cand_html = (
            f"<b>NPC:</b> <pre style='white-space:pre-wrap'>{npc}</pre>\n"
            f"<b>Base:</b> <pre style='white-space:pre-wrap'>{base}</pre>\n"
            f"<b>Secondary:</b> <pre style='white-space:pre-wrap'>{secondary}</pre>\n"
            f"<b>MPC:</b> <pre style='white-space:pre-wrap'>{mpc}</pre>\n"
            f"<b>EAN/GTIN:</b> <pre style='white-space:pre-wrap'>{ean}</pre>\n"
        )

        return manuf_html, candidates_html, device_html, device_cand_html

    def _show(self):
        i = self.idx
        manuf_html, candidates_html, device_html, device_cand_html = self._render_row(i)

        self.manuf_label.value = f"<b>Row {i+1} / {self.N}</b><br>{manuf_html}"
        self.manuf_candidates.value = candidates_html
        self.device_label.value = device_html
        self.device_candidates.value = device_cand_html

    def _save_current(self):
        # Save current widget values into the sparse results dict
        i = self.idx
        entry = {
            'manufacturer_match': self.manuf_choice.value,
            'manufacturer_comment': self.manuf_comment.value,
            'device_match': self.device_choice.value,
            'device_comment': self.device_comment.value,
        }
        # If all values are empty/None, remove existing entry to keep storage small
        if all(v in (None, '', []) for v in entry.values()):
            self.results.pop(i, None)
        else:
            self.results[i] = entry

    def _on_prev(self, _):
        self._save_current()
        if self.idx > 0:
            self.idx -= 1
        self._show()
        if self.autosave:
            self._autosave()

    def _on_next(self, _):
        self._save_current()
        if self.idx < self.N - 1:
            self.idx += 1
        self._show()
        if self.autosave:
            self._autosave()

    def _on_jump(self, change):
        new = int(change['new']) - 1
        if 0 <= new < self.N:
            self._save_current()
            self.idx = new
            self._show()

    def _on_save(self, _):
        self._save_current()
        self._autosave()

    def _on_export(self, _):
        self._save_current()
        # export only reviewed rows by default (fast)
        if self.export_reviewed_only:
            if not self.results:
                with self.out:
                    print('No reviewed rows to export.')
                return
            idxs = sorted(self.results.keys())
            out_df = self.df.loc[idxs].copy()
            # add review columns from sparse dict
            out_df['manufacturer_match'] = [self.results[i]['manufacturer_match'] for i in idxs]
            out_df['manufacturer_comment'] = [self.results[i]['manufacturer_comment'] for i in idxs]
            out_df['device_match'] = [self.results[i]['device_match'] for i in idxs]
            out_df['device_comment'] = [self.results[i]['device_comment'] for i in idxs]
        else:
            # build a full annotated copy (slower)
            out_df = self.df.copy()
            out_df['manufacturer_match'] = None
            out_df['manufacturer_comment'] = None
            out_df['device_match'] = None
            out_df['device_comment'] = None
            for i, vals in self.results.items():
                out_df.at[i, 'manufacturer_match'] = vals.get('manufacturer_match')
                out_df.at[i, 'manufacturer_comment'] = vals.get('manufacturer_comment')
                out_df.at[i, 'device_match'] = vals.get('device_match')
                out_df.at[i, 'device_comment'] = vals.get('device_comment')

        out_df.to_csv(self.filename, index=False)
        with self.out:
            print(f"Exported reviewed sample to {self.filename}")

    def _autosave(self):
        # quick save only reviewed rows to a progress file (fast)
        if not self.results:
            return
        tmpname = self.filename.replace('.csv', '.progress.csv')
        idxs = sorted(self.results.keys())
        out_df = self.df.loc[idxs].copy()
        out_df['manufacturer_match'] = [self.results[i]['manufacturer_match'] for i in idxs]
        out_df['manufacturer_comment'] = [self.results[i]['manufacturer_comment'] for i in idxs]
        out_df['device_match'] = [self.results[i]['device_match'] for i in idxs]
        out_df['device_comment'] = [self.results[i]['device_comment'] for i in idxs]
        out_df.to_csv(tmpname, index=False)

    def get_reviewed_dataframe(self, only_reviewed=True):
        """Return the reviewed rows merged with review columns.

        Parameters
        ----------
        only_reviewed : bool
            If True (default) return only the reviewed rows with annotations.
            If False, return the full dataset with annotation columns (may be slow).
        """
        if only_reviewed:
            if not self.results:
                return pd.DataFrame(columns=list(self.df.columns) + ['manufacturer_match','manufacturer_comment','device_match','device_comment'])
            idxs = sorted(self.results.keys())
            out_df = self.df.loc[idxs].copy()
            out_df['manufacturer_match'] = [self.results[i]['manufacturer_match'] for i in idxs]
            out_df['manufacturer_comment'] = [self.results[i]['manufacturer_comment'] for i in idxs]
            out_df['device_match'] = [self.results[i]['device_match'] for i in idxs]
            out_df['device_comment'] = [self.results[i]['device_comment'] for i in idxs]
            return out_df
        else:
            # full annotated copy
            full = self.df.copy()
            full['manufacturer_match'] = None
            full['manufacturer_comment'] = None
            full['device_match'] = None
            full['device_comment'] = None
            for i, vals in self.results.items():
                full.at[i, 'manufacturer_match'] = vals.get('manufacturer_match')
                full.at[i, 'manufacturer_comment'] = vals.get('manufacturer_comment')
                full.at[i, 'device_match'] = vals.get('device_match')
                full.at[i, 'device_comment'] = vals.get('device_comment')
            return full
        
class DeviceLabeler:
    """
    Interactive labeling widget for unlabeled rows in a DataFrame.
    - df: DataFrame with rows to label (works on a working copy)
    - cols_to_show: list of columns to display for context
    - label_col: name of the column to store golden labels (created if missing)
    - index_col: column name used to store original index to merge back if needed
    - output_path: where `save_all()` will write CSV
    - shuffle / random_state: optionally shuffle the working copy reproducibly
    """
    def __init__(self, df, cols_to_show=None, label_col='golden_label',
                 index_col='orig_index', output_path='data/outputs/df_devices_nulls_labeled.csv',
                 shuffle=True, random_state=42):
        self.label_col = label_col
        self.index_col = index_col
        self.output_path = output_path

        # Make a working copy and ensure label and original-index columns
        self._df = df.copy().reset_index(drop=True)
        if self.label_col not in self._df.columns:
            self._df[self.label_col] = pd.NA
        if self.index_col not in self._df.columns:
            self._df[self.index_col] = df.index

        # Optional shuffle for review order
        if shuffle:
            self._df = self._df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # default columns to show if not provided
        if cols_to_show is None:
            cols_to_show = ['CLN_Manufacturer', 'Supplier_cat', 'Brand',
                            'CLN_Manufacturer_Device_Name', 'matched_device', 'device_level']
        self.cols_to_show = [c for c in cols_to_show if c in self._df.columns]

        # State
        self.unlabeled = self._df.index[self._df[self.label_col].isna()].tolist()
        self.pos = 0

        # Widgets
        self.out = widgets.Output()
        self.label_input = widgets.Text(placeholder='Type label (e.g. device name or code)', description='Label:')
        self.save_btn = widgets.Button(description='Save label', button_style='success')
        self.unlab_btn = widgets.Button(description='Mark unlabelable', button_style='warning')
        self.prev_btn = widgets.Button(description='Previous')
        self.next_btn = widgets.Button(description='Next')
        self.save_all_btn = widgets.Button(description='Save all to CSV')
        self.status = widgets.Label()

        # Attach handlers
        self.save_btn.on_click(self._save_label)
        self.unlab_btn.on_click(self._mark_unlabelable)
        self.prev_btn.on_click(self._go_prev)
        self.next_btn.on_click(self._go_next)
        self.save_all_btn.on_click(self._save_all)

    # Internal helpers
    def _update_controls(self):
        n = len(self.unlabeled)
        self.save_btn.disabled = (n == 0)
        self.unlab_btn.disabled = (n == 0)
        self.prev_btn.disabled = (n == 0 or self.pos == 0)
        self.next_btn.disabled = (n == 0 or self.pos >= n - 1)
        self.status.value = f"Unlabeled remaining: {n}"

    def _show_current(self):
        with self.out:
            clear_output()
            n = len(self.unlabeled)
            if n == 0:
                display(HTML("<b>All rows are labeled.</b>"))
                return
            i = self.unlabeled[self.pos]
            row = self._df.loc[i]
            display(HTML(f"<b>Unlabeled {self.pos+1}/{n} — original index {row[self.index_col]}</b>"))
            if self.cols_to_show:
                display(row[self.cols_to_show].to_frame().T)
            self.label_input.value = '' if pd.isna(row.get(self.label_col)) else str(row.get(self.label_col))

    # Handlers
    def _save_label(self, _=None):
        if not self.unlabeled:
            return
        i = self.unlabeled[self.pos]
        val = self.label_input.value.strip()
        self._df.at[i, self.label_col] = pd.NA if val == '' else val
        self.unlabeled.pop(self.pos)
        if self.pos >= len(self.unlabeled) and self.pos > 0:
            self.pos -= 1
        self._update_controls()
        self._show_current()

    def _mark_unlabelable(self, _=None):
        if not self.unlabeled:
            return
        i = self.unlabeled[self.pos]
        self._df.at[i, self.label_col] = 'UNLABELABLE'
        self.unlabeled.pop(self.pos)
        if self.pos >= len(self.unlabeled) and self.pos > 0:
            self.pos -= 1
        self._update_controls()
        self._show_current()

    def _go_prev(self, _=None):
        if self.pos > 0:
            self.pos -= 1
        self._update_controls()
        self._show_current()

    def _go_next(self, _=None):
        if self.pos < len(self.unlabeled) - 1:
            self.pos += 1
        self._update_controls()
        self._show_current()

    def _save_all(self, _=None):
        self._df.to_csv(self.output_path, index=False)
        print(f"Saved current labeled df to {self.output_path}")

    # Public API
    def display(self):
        """Render the widget in the notebook."""
        controls = widgets.HBox([self.prev_btn, self.next_btn, self.save_all_btn])
        actions = widgets.HBox([self.label_input, self.save_btn, self.unlab_btn])
        self._update_controls()
        display(widgets.VBox([controls, actions, self.status, self.out]))
        self._show_current()

    def get_labeled_df(self):
        """Return the working copy with labels applied."""
        return self._df.copy()

    def merge_back_to(self, original_df, on=None, how='left'):
        """
        Merge labeled column back into the original dataframe.
        - If `on` is None, it uses the `index_col` (orig_index) to merge.
        """
        if on is None:
            on = self.index_col
            return original_df.merge(self._df[[self.index_col, self.label_col]],
                                     left_on=original_df.index, right_on=self.index_col,
                                     how=how, suffixes=('', '_new')).drop(columns=['key_0'], errors='ignore')
        else:
            return original_df.merge(self._df[[on, self.label_col]], on=on, how=how)
