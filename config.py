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
# recent signal on the chart. "none" labels it NO TRADE, because the signals
# conflict and there is no clean trade. I chose "none": a trader who sees both
# a bullish and a bearish pattern on one chart has no clean trade. I tested
# "latest" too and report it as the alternative.
MIXED_CHART_RULE = "none"

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
# DECISION ECONOMICS
# --------------------------------------------------------------------------
# The desk: an end-of-day screen at a long/short equity fund. After the close
# the model scans one full-day chart per stock. Flagged trades go on at the
# next open, and an analyst reviews the unclear charts in the evening.
#
# One case = one chart = one stock on one day. For each chart the system does
# one of four things: trade LONG, trade SHORT, NO TRADE, or send it to an
# analyst (REVIEW).
#
# Costs are in euros per chart, measured against a perfect trader who always
# gets the call right, so a correct call costs 0.
#
# Figures proposed by Claude from my inputs and common fund practice, then
# reviewed and approved by me. Every one is tested in the dashboard's
# Sensitivity tab.

COSTS_ARE_PLACEHOLDERS = False

BOOK_EUR = 100_000_000      # a mid-sized long/short fund
POSITION_SHARE = 0.005      # 0.5% of the book per pattern trade: a tactical
                            # position, well below a fundamental core holding
POSITION_EUR = BOOK_EUR * POSITION_SHARE          # 500,000 euros per trade

EXPECTED_MOVE = 0.005       # 0.5% next-day move captured on a real pattern,
                            # before costs. Evidence for chart patterns is thin
                            # (Lo, Mamaysky and Wang 2000, Neely 2003), so this
                            # is deliberately modest
ROUND_TRIP_COST = 0.0020    # 20 bps in and out: liquid large caps, modest size

# A trade on a chart with no pattern is a coin flip, so on average it loses
# only the trading costs. The perfect trader trades a real pattern only if the
# move beats the costs, so each cost is measured against that best action.
NET_EDGE = max(EXPECTED_MOVE - ROUND_TRIP_COST, 0.0)   # what the perfect trader earns, as a share
COST_MISSED_SETUP = POSITION_EUR * NET_EDGE            # profit forgone: 1,500 euros
COST_SPURIOUS_TRADE = POSITION_EUR * ROUND_TRIP_COST   # costs paid on noise: 1,000 euros
COST_WRONG_DIRECTION = POSITION_EUR * (NET_EDGE + EXPECTED_MOVE + ROUND_TRIP_COST)
                                                       # forgo the edge, lose the move, pay costs: 5,000 euros
COST_CORRECT_TRADE = POSITION_EUR * max(ROUND_TRIP_COST - EXPECTED_MOVE, 0.0)
                                                       # 0 unless the move is below costs
COST_CORRECT_PASS = 0.0

# Share of charts that really contain a tradable pattern in production.
# 0.57 is the share of training-day charts with a clean, single-direction
# pattern (465 of 814): the largest sample, and it never uses the test days.
# None would use the test set's own share instead.
BASE_RATE = 0.57

# The review desk.
CHARTS_PER_DAY = 500                 # a large-cap US universe, one chart per stock
MINUTES_PER_REVIEW = 2.0             # look at the chart, check for news, decide
ANALYST_COST_PER_HOUR = 60_000 * 1.5 / 2_000   # 60k salary, 1.5x fully loaded,
                                               # 2,000 working hours = 45 euros
REVIEW_HOURS_PER_ANALYST_PER_DAY = 1.0  # one evening hour each, beside other work
MAX_ANALYSTS = 2.0
REVIEWER_ACCURACY = 0.9              # chart patterns are subjective, so two people
                                     # will disagree on some. A wrong call counts as
                                     # a missed setup on a real pattern and as a
                                     # spurious trade on a clean chart.
