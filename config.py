"""
Every assumption the pipeline and the dashboard depend on, in one place.

train.py and app.py both read this file, so changing a number here and
re-running `main.py app` updates the dashboard. Re-run `main.py train` only if
you change something in the DATA or TRAINING sections.

The DECISION ECONOMICS section holds placeholders. They exist so the pipeline
runs end to end. Replace every one of them with your own figure and justify it
in REPORT.md, then set COSTS_ARE_PLACEHOLDERS = False.
"""

# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------

ROBOFLOW_WORKSPACE = "stock-patterns"
ROBOFLOW_PROJECT = "stocks-3patterns-2m-sept2023"
ROBOFLOW_VERSION = 1

# The published version is 100% train, so we make our own split, by trading
# date read from the file name (e.g. "uber_2m_2023-09-18_to_2023-09-19.png").
# The last TEST_DATES dates are the test set, the VALID_DATES before them are
# validation, everything earlier is training. Splitting by date means the model
# is always scored on days it has never seen, which is how it would be used.
TEST_DATES = 2
VALID_DATES = 2

# Which way each pattern says to trade. Names are matched case-insensitively,
# with "-" and "_" treated as spaces.
CLASS_SIDE = {
    "double bottom": "LONG",
    "double top": "SHORT",
    "head and shoulders": "SHORT",
}

# Some charts carry both a bullish and a bearish pattern. "latest" labels such
# a chart by the pattern that finishes furthest to the right, i.e. the most
# recent signal on the chart.
MIXED_CHART_RULE = "latest"

# --------------------------------------------------------------------------
# TRAINING
# --------------------------------------------------------------------------

BASE_MODEL = "yolo11s.pt"  # finetuned from this COCO checkpoint
EPOCHS = 100
PATIENCE = 25  # stop early if validation mAP has not improved for this many epochs
IMGSZ = 640
BATCH = 16
SEED = 42

# Augmentation. Horizontal flips are off because they reverse time: a flipped
# chart is not a chart a trader would ever see. Vertical flips (off by default
# in Ultralytics) would turn a double top into a double bottom.
FLIPLR = 0.0
FLIPUD = 0.0

# Confidence floor used when saving predictions. Kept very low so the whole
# threshold range can be swept afterwards without re-running the model.
PREDICT_CONF_FLOOR = 0.001
MAX_BOXES_SAVED_PER_CHART = 20

# --------------------------------------------------------------------------
# DECISION ECONOMICS  (placeholders: replace every figure with your own)
# --------------------------------------------------------------------------
# One case = one chart = one stock on one day. For each chart the system does
# one of four things: trade LONG, trade SHORT, NO TRADE, or send it to an
# analyst (REVIEW).
#
# Costs are in euros per chart and measured against a perfect trader who
# always gets the call right, so a correct call costs 0 by default.

COSTS_ARE_PLACEHOLDERS = True

COST_MISSED_SETUP = 300.0      # a real pattern, the system says NO TRADE
COST_SPURIOUS_TRADE = 600.0    # no pattern, the system trades anyway
COST_WRONG_DIRECTION = 1500.0  # a real pattern, the system trades the opposite side
COST_CORRECT_TRADE = 0.0       # a real pattern, traded the right way
COST_CORRECT_PASS = 0.0        # no pattern, NO TRADE

# Share of charts that really contain a tradable pattern in production.
# None means "use the share in the test set", which overstates it if the
# dataset was built around charts that had patterns.
BASE_RATE = None

# The review desk.
CHARTS_PER_DAY = 500                 # names screened each day, one chart each
MINUTES_PER_REVIEW = 1.0             # analyst time to check one flagged chart
ANALYST_COST_PER_HOUR = 60.0         # fully loaded, in euros
REVIEW_HOURS_PER_ANALYST_PER_DAY = 2.0  # hours a day an analyst can spend on this
MAX_ANALYSTS = 1.0                   # people actually available for the queue
REVIEWER_ACCURACY = 1.0              # chance an analyst makes the right call on a reviewed chart.
                                     # A wrong call counts as a missed setup on a real pattern
                                     # and as a spurious trade on a clean chart.
