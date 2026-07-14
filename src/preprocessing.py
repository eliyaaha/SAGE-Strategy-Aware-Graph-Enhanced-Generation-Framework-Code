import re
import json
import pandas as pd
import numpy as np

def clean_text(s: str) -> str:
    """Minimal whitespace and character cleanup."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_neutral_label(df, neutral_label="ניטרלית"):
    """
    Removes the neutral label from the 'processed' column.
    - If it's the only label on a row, the row becomes NaN (no strategy).
    - If combined with real strategies (e.g. "שיקוף + ניטרלית"), only the
      neutral token is dropped, keeping the remaining strategies joined by " + ".
    """
    def _clean(val):
        if pd.isna(val):
            return val
        parts = [p.strip() for p in str(val).split("+") if p.strip()]
        kept = [p for p in parts if p != neutral_label]
        return " + ".join(kept) if kept else np.nan

    df["processed"] = df["processed"].apply(_clean)
    return df

def load_and_merge_data(raw_path, blocks_path):
    """
    Load raw annotations and blocked conversations, then merge them on engagement_id.
    """
    # Load files
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    df_blocks = pd.read_csv(blocks_path, encoding="utf-8-sig")

    # Standardize engagement_id format
    df_blocks["engagement_id"] = df_blocks["engagement_id"].astype(str).str.strip()
    df_raw["engagement_id"] = df_raw["engagement_id"].astype(str).str.strip()

    # Define metadata columns to extract
    metadata_cols = [
        'date_conv', 'name_conv', 'gender', 'age', 'ved',
        'gsr', 'imsr', 'subject_1', 'subject_2', 'subject_3',
        'subject_4', 'subject_5', 'subject_6', 'date_msg', 'time'
    ]

    # Group metadata by conversation ID, taking the first occurrence
    df_meta = df_raw.groupby("engagement_id")[metadata_cols].first().reset_index()

    # Merge metadata into the blocked message dataframe
    df_merged = df_blocks.merge(df_meta, on="engagement_id", how="left")
    
    return df_merged

def apply_lexicon_matching(df, lexicon_path):
    """
    Enrich the dataframe with binary columns based on a clinical lexicon.
    Matches phrases from 'gsr' categories specifically in seeker messages.
    """
    with open(lexicon_path, "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    
    # Focusing only on 'gsr' category as per the latest requirements
    top_categories = ["gsr"]
    category_to_phrases = {}
    lexicon_columns = []

    # Map subcategories to phrase sets and initialize columns
    for top in top_categories:
        subcats = lexicon.get(top, {})
        for subcat, phrases in subcats.items():
            combined_name = f"{top}_{subcat}"
            category_to_phrases[combined_name] = list(set(phrases))
            lexicon_columns.append(combined_name)
            df[combined_name] = 0  # Initialize with zeros

    # Perform matching only for seeker messages
    is_seeker = df["seeker"] == True
    
    for combined_cat, phrase_list in category_to_phrases.items():
        # Iterate only through seeker rows for efficiency
        for idx, row in df[is_seeker].iterrows():
            text = str(row["text"])
            for phrase in phrase_list:
                if phrase in text:
                    df.at[idx, combined_cat] = 1
                    break # Stop after the first match for this category in this message
                    
    return df, lexicon_columns

def process_labels(df, target_labels):
    """Parse and filter strategy labels from the raw CSV format."""
    def split_labels(l):
        return [p.strip() for p in str(l).split("+") if p.strip()]

    df["labels_list_msg"] = df["processed"].apply(
        lambda x: [l for l in split_labels(x) if l in target_labels] if pd.notna(x) else []
    )
    return df

def create_expert_seeker_pairs(df):
    """Build pairs of seeker messages and the subsequent expert therapeutic response."""
    # Ensure chronological order within each conversation
    df = df.sort_values(["engagement_id", "message_id"]).reset_index(drop=True)
    
    # Identify the most recent seeker message for each turn
    df["seeker_text_candidate"] = np.where(df["seeker"], df["text"], np.nan)
    df["last_seeker_text"] = df.groupby("engagement_id")["seeker_text_candidate"].ffill()
    
    # Filter for expert messages that have valid labels and a preceding seeker message
    valid_expert = (~df["seeker"]) & (df["labels_list_msg"].map(len) > 0) & (df["last_seeker_text"].notna())
    
    pairs = pd.DataFrame({
        "text": df.loc[valid_expert, "last_seeker_text"].astype(str).apply(clean_text).values,
        "labels_list": df.loc[valid_expert, "labels_list_msg"].values,
        "df_idx": np.where(valid_expert)[0]
    })
    
    return pairs.dropna(subset=["text"])

def collapse_consecutive_turns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3 of the original pipeline (merge_therapist_blocks):
    merges consecutive therapist (seeker == False) rows within the same
    engagement_id. Seeker rows are never merged here (already block-merged
    upstream in blocked_150_conv.csv / Stage 1).
    """
    df = df.sort_values(["engagement_id", "message_id"]).reset_index(drop=True)

    merged_rows = []
    current_eng = None
    therapist_block_active = False

    buffer_text = []
    buffer_processed_set = set()
    buffer_processed_list = []
    buffer_ids = []
    buffer_row_template = None

    def flush_therapist_block():
        nonlocal therapist_block_active
        if not therapist_block_active:
            return
        new_row = buffer_row_template.copy()
        new_row["text"] = "\n".join(buffer_text)
        new_row["processed"] = " + ".join(buffer_processed_list) if buffer_processed_list else ""
        new_row["message_id"] = buffer_ids[0]
        merged_rows.append(new_row)

        therapist_block_active = False
        buffer_text.clear()
        buffer_processed_set.clear()
        buffer_processed_list.clear()
        buffer_ids.clear()

    for _, row in df.iterrows():
        eng = row["engagement_id"]
        is_seeker = bool(row["seeker"])

        if current_eng is not None and eng != current_eng:
            flush_therapist_block()
        current_eng = eng

        if is_seeker:
            flush_therapist_block()
            merged_rows.append(row.to_dict())
            continue

        if not therapist_block_active:
            therapist_block_active = True
            buffer_row_template = row.to_dict()

        buffer_text.append("" if pd.isna(row["text"]) else str(row["text"]))

        proc = row["processed"]
        if pd.notna(proc):
            for p in re.split(r"\s*\+\s*", str(proc)):
                p = p.strip()
                if p and p != "ניטרלית" and p not in buffer_processed_set:
                    buffer_processed_set.add(p)
                    buffer_processed_list.append(p)

        buffer_ids.append(row["message_id"])

    flush_therapist_block()

    return pd.DataFrame(merged_rows)