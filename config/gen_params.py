import torch

# --- Model Selection  ---
GEMMA_MODEL_NAME = "google/gemma-3-12b-it"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SAGE / Injection Hyperparameters  ---
GA_LR_LORA = 5e-6           
GA_LR_GRAPH = 1e-4          # For the projection and attention layers
INITIAL_GRAPH_GATING = 0.5  # Controls the initial graph influence
NUM_VIRTUAL_TOKENS = 8      # Number of soft prompt tokens
EPOCHS_GRAPHAWARE = 7

# --- Generation Constraints  ---
MAX_TURNS = 5               # Context history turns
MAX_NEW_TOKENS = 60         # Maximum length of therapeutic response
GEN_TEMPERATURE = 0.4       # For balanced creativity and consistency
GEN_TOP_P = 0.9             # Nucleus sampling
GEN_REP_PENALTY = 1.15      # To avoid repetitive phrasing
NO_REPEAT_NGRAM = 3         # Block repeating phrases

# --- Strategy Definitions ---
STRATEGY_DEFINITIONS = { "שיקוף" : "מתן תוקף לתחושת הפונה, חזרה על דברי הפונה תוך הכלה ואמפתיה.\n"
                                    "הדהוד התוכן שלו במילים אחרות שיעידו על הקשבה והבנה של הטקסט.",
                        "דיבוב" : "הזמנה להרחיב או לספק מידע נוסף, תגובות שמבהירות או שואלות על מה שנאמר.\n"
                                    "במידה ועולה תוכן אובדני, יש להביע דאגה ולשאול אם יש כוונה למות הלילה.",
                        "מתן נקודה למחשבה" : "הצעה זהירה וצנועה לעצה, כיוון מחשבה או פעולה, עידוד לפעולה או להסתכלות חדשה.\n"
                                    "להזכיר שהמצב זמני, ולזהות נקודות חוסן (למשל בן משפחה או חבר) שיוכלו לסייע."}

# --- Prompt Construction Constants ---
SYSTEM_INSTRUCTION = "אתה מטפל מתחום בריאות הנפש בצ'אט לייעוץ נפשי בשפה העברית."
CLINICAL_GUIDELINE_HEADER = "הנחיות קליניות לניסוח התגובה:\nעל סמך ניתוח המצב הרגשי של הפונה, עליך לשלב בתשובתך את הטכניקות הטיפוליות הבאות:"
CLINICAL_GUIDELINE_FOOTER = "דגש חשוב: אל תציין את שמות הטכניקות בתשובתך, אלא יישם אותן באופן טבעי בתוך השיחה."