import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from config.gen_params import STRATEGY_DEFINITIONS, MAX_TURNS

def build_history_text(df, conv_id, current_msg_idx, max_turns=MAX_TURNS):
    """
    Constructs a string of the last N turns before the current message.
    """
    # Filter messages from the same conversation that appeared before the current one
    sub = df[(df["engagement_id"] == conv_id) & (df.index < current_msg_idx)]
    
    blocks = []
    for _, r in sub.iterrows():
        role = "פונה" if r["seeker"] else "מטפל"
        blocks.append(f"{role}: {r['text']}")
        
    # Keep only the last MAX_TURNS to avoid context overflow
    if len(blocks) > max_turns:
        blocks = blocks[-max_turns:]
        
    return "\n".join(blocks)

def build_prompt_ft(history_text, seeker_text):
    """
    Prompt aligned exactly with FT training format (without strategy targets).
    """
    return (
        "אתה מטפל מתחום בריאות הנפש בצ'אט לייעוץ נפשי בשפה העברית.\n"
        "להלן היסטוריית הצ'אט:\n"
        f"{history_text}\n\n"
        "להלן הודעת הפונה האחרונה:\n"
        f"פונה: {seeker_text}\n\n"
        "עלייך לנסח תגובה טיפולית קצרה, במשפט אחד או שניים, תומכת וברורה בעברית, " 
        "שמתייחסת ישירות לתוכן שכתב הפונה.\n\n"
        "מטפל:"
    )

def build_prompt_ft_with_strategy(history_text, seeker_text, strategy_labels):
    """
    Constructs a strategy-aware prompt including clinical definitions.
    Handles multiple strategies joined by '+'.
    """
    strategy_block = ""
    
    if strategy_labels:
        # Split string labels if they are joined by '+'
        if isinstance(strategy_labels, str):
            labels = [s.strip() for s in strategy_labels.split("+")]
        else:
            labels = strategy_labels
        
        strategy_items = []
        for name in labels:
            definition = STRATEGY_DEFINITIONS.get(name, "אין הגדרה זמינה")
            strategy_items.append(f"{name} - {definition}")
        
        # Combine all name-definition pairs
        combined_strategies = "\n".join(strategy_items)
        
        strategy_block = (
            "הנחיות קליניות לניסוח התגובה:\n"
            "על סמך ניתוח המצב הרגשי של הפונה, עליך לשלב בתשובתך את הטכניקות הטיפוליות הבאות:\n"
            f"{combined_strategies}\n\n"
            "דגש חשוב: אל תציין את שמות הטכניקות בתשובתך, אלא יישם אותן באופן טבעי בתוך השיחה.\n\n"
        )

    return (
        "אתה מטפל מתחום בריאות הנפש בצ'אט לייעוץ נפשי בשפה העברית.\n"
        "להלן היסטוריית הצ'אט:\n"
        f"{history_text}\n\n"
        f"{strategy_block}"
        "להלן הודעת הפונה האחרונה:\n"
        f"פונה: {seeker_text}\n\n"
        "עלייך לנסח תגובה טיפולית קצרה, במשפט אחד או שניים, תומכת וברורה בעברית, "
        "שמתייחסת ישירות לתוכן שכתב הפונה.\n\n"
        "מטפל:"
    )