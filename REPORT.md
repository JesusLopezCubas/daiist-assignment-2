# Assignment 2 Report: a chart pattern screen for a long/short equity desk

- **Name**: Jesús López Cubas
- **Student ID**: 20414
- **Email**: jlopez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

**The decision.** A long/short equity fund with a €100m book runs an end of day screen over 500 liquid US large caps. After the close, the model reads one chart per stock: the full trading day in 2 minute candles. For each chart the desk does one of four things. It opens a long at the next open, opens a short, does nothing, or puts the chart in an evening queue for an analyst to decide. Today a trader makes this call by eye, which limits coverage to the handful of names a person can look at. The model's job is triage: act on the clear charts and send only the uncertain ones to a person.

I chose an end of day screen rather than a live intraday scanner because it matches the data exactly. Each image in the dataset is a complete day, so the model sees what the desk sees after the close and nothing from the future. A live scanner would only see part of the day, and the results below would not hold for it.

**The positive class** is a chart with a tradable pattern that points one way. A double bottom says go long. A double top or a head and shoulders says go short. Two kinds of chart are "no trade": charts with no pattern, and charts that carry both a bullish and a bearish pattern. A trader who sees conflicting signals on the same chart has no clean trade, so he does not trade it.

**What each mistake costs.** Every cost is measured against a perfect trader who always makes the right call, so a correct call costs nothing. The inputs:

- **Position size €500,000**, which is 0.5% of the book. A short term technical signal is a tactical position, well below the size of a fundamental core holding.
- **Expected move 0.5%** on the next day for a real pattern traded the right way. I kept this deliberately modest because the evidence that chart patterns pay is thin. Lo, Mamaysky and Wang (2000), studying US stocks from 1962 to 1996, found that some patterns carry incremental information that "may have some practical value", which is hardly a strong edge. Neely (2003) found that intraday trading rules in currencies earn nothing after realistic costs.
- **Trading costs of 20 bps** for a round trip. Nasdaq's analysis of buy side data puts the average institutional cost at about 31.5 bps of spread and market impact per trade, and the names in this universe are among the most liquid there are, traded in modest size.
- **A trade on a chart with no real pattern is a coin flip.** On average it gains nothing and pays the trading costs.

| Outcome | Cost per chart | How it is built |
|---|---|---|
| Missed setup: a real pattern, the desk does not trade | €1,500 | The net edge forgone: €500,000 × (0.5% − 0.2%) |
| Spurious trade: no clean pattern, the desk trades | €1,000 | The trading costs paid on noise: €500,000 × 0.2% |
| Wrong direction: a real pattern, traded the opposite way | €5,000 | Forgo the €1,500 edge, lose the €2,500 move and pay €1,000 of costs |
| Correct trade or correct pass | €0 | The benchmark |
| Chart sent to an analyst | €1.50, plus the cost of an analyst error | 2 minutes at €45 an hour, with the analyst right 9 times in 10 |

The analyst rate is a €60,000 salary, times 1.5 for bonus, social charges and desk costs, over 2,000 working hours: €45 an hour. When the analyst gets a reviewed chart wrong, I count it as a missed setup on a real pattern and as a spurious trade on a clean chart.

**The asymmetry.** Trading a real pattern the wrong way is by far the worst mistake: it costs 3.3 times a missed setup and 5 times a spurious trade. Missing a real setup costs 1.5 times trading on noise, because with a 0.5% edge the profit forgone is larger than the costs paid on a coin flip. This ordering drives everything that follows. The metric that matters is the euro cost per chart and whether the model points the right way, not mAP on its own. The model has to be confident before it trades, because a confident wrong call is the expensive one. And the analysts are best spent on the charts where the direction is uncertain.

## Dataset

- **Roboflow Universe**: workspace `stock-patterns`, project `stocks-3patterns-2m-sept2023`, version `1`
- **Link**: https://universe.roboflow.com/stock-patterns/stocks-3patterns-2m-sept2023/dataset/1
- **Task**: object detection with three classes, `double bottom`, `double top` and `head and shoulders`. The background is the free fourth class.
- **Licence**: CC BY 4.0

One image is one ticker on one trading day, drawn from Yahoo Finance data in 2 minute candles. The dataset has 1,110 images covering 74 tickers over 15 trading days, from 11 to 29 September 2023. 71 of the tickers are US stocks and 3 are currency pairs (CAD=X, EURCAD=X and EURUSD=X).

**Why it serves the problem rather than merely resembling it.** Each image is exactly the picture the desk looks at after the close. Each class maps directly to a trade direction. And because it is a detection dataset, every pattern comes with its location. That matters twice. It lets the system see when one chart holds two patterns pointing opposite ways, which the "no trade" rule depends on. And the box tells the analyst where to look on a chart sent for review. An image classification dataset would give one label per chart and lose both.

**The split.** The published version puts all 1,110 images in the training split, with no validation or test data. I used it as published and split it in code by trading date: the first 11 days for training, the next 2 for validation and the last 2 for testing. Charts from the same day share market wide moves, so a random split would place closely related charts on both sides and flatter the test results. Splitting by date means the model is always scored on days it has never seen, which is how it would be used. The split is deterministic, so anyone who runs the pipeline gets the same one.

## What the dataset's composition implies

| Split | Days | Charts | No pattern | Both directions | Clean single direction pattern | Double bottom boxes | Double top boxes | Head and shoulders boxes |
|---|---|---|---|---|---|---|---|---|
| Train | 11 | 814 | 183 (22%) | 166 (20%) | 465 (57%) | 338 | 557 | 94 |
| Valid | 2 | 148 | 27 (18%) | 56 (38%) | 65 (44%) | 75 | 136 | 22 |
| Test | 2 | 148 | 15 (10%) | 36 (24%) | 97 (66%) | 66 | 127 | 19 |

A typical pattern box covers about 2% of the chart's area and 10% of its width. The smallest tenth are around 5% of the width.

What this means for the results:

1. **Head and shoulders is too thin to trust.** The test days hold only 19 of them. At my operating threshold the model finds 3, and with 19 cases the 95% interval for that recall runs from about 6% to 38%. I draw no conclusion about head and shoulders from this test set.
2. **The classes are unbalanced towards double tops**, which are 56% of all training boxes. I expect the model to lean bearish, and it does (see the failure modes).
3. **Patterns are everywhere.** Only 10% to 22% of charts have no pattern at all, which suggests the labeller marked loose patterns. The share of charts with a pattern that really pays is probably lower in production, so I test the base rate in the sensitivity analysis rather than trust one number.
4. **Conflicting charts are common**, 20% to 38% depending on the split. How they are labelled moves the results a lot, so I tested both rules (see the cost section).
5. **The three splits look different from each other.** Clean patterns are 44% of the validation charts but 66% of the test charts. Two days is a small sample, so I do not cost each split at its own mix. I weight every result to one fixed base rate taken from the training days.
6. **The same 74 tickers appear every day.** The test measures new days for known names, not new names. The desk's universe of 500 stocks is larger, so generalisation to unseen stocks is untested.
7. **One market regime.** Two weeks of September 2023 say nothing about how the model behaves in a sell off or a high volatility spell.
8. **Small objects.** At 640 pixels, a box 10% of the width is about 60 pixels wide, and the narrowest are about 30. Resizing a full day of candles to 640 pixels loses detail, which is one reason the model misses patterns.
9. **Three tickers are currencies**, not stocks. I kept them so the public dataset stays as published. They are 4% of the charts.

## Finetuning and baselines

**Model.** YOLO11s for object detection, finetuned from the COCO checkpoint. I chose detection over classification because one chart can hold several patterns and the location is needed, as explained above. I chose the "s" model over the "n" because it is the larger of the two allowed, and training is cheap on a GPU.

**Settings.** Image size 640, batch 16, up to 100 epochs with early stopping after 25 epochs without improvement, seed 42. Horizontal and vertical flips are off: a horizontal flip runs time backwards and a vertical flip turns a double top into a double bottom, so neither creates a chart a trader would ever see. Everything else is left at the Ultralytics defaults.

**Where I trained.** On a Google Colab T4 GPU, with `uv run python main.py train`. Training stopped after 43 epochs, kept epoch 18 (the best validation mAP50-95) and took 10.5 minutes. I ran it three times and got identical metrics each time, so the result is reproducible. On validation the best epoch scored mAP50 0.533 and mAP50-95 0.361. On the test days it scores 0.402 and 0.272. A drop is expected, because the epoch was chosen on validation and the test days have a different mix of charts.

**Results on the same 148 test charts.** Each model's confidence threshold is chosen on the validation days by the cost sweep described below.

| Model | mAP50 (3 classes) | Finds a pattern at all (class agnostic AP50) | Threshold | Real patterns traded (recall) | Trades on a real pattern (precision) | Expected cost per 1,000 charts |
|---|---|---|---|---|---|---|
| **Finetuned YOLO11s** | **0.402** | **0.501** | 0.60 | **43%** | 59% | **€758,685** |
| COCO YOLO11s, zero-shot | 0 | 0 | 0.06 | 2% | 67% | €845,803 |
| Predict nothing (also the majority class) | 0 | 0 | none | 0% | none | €855,000 |

The class agnostic AP50 asks only whether a box lands on a pattern, ignoring which pattern. It is the only box metric on which the COCO model can be compared, since COCO has no chart classes. Zero-shot, COCO's most confident box on a chart is usually a "person" (91 of the 148 charts) or "skis" (32), with a median confidence of zero. The most common chart label in training is "no trade" (349 of 814 charts), so the majority class baseline and the predict nothing baseline are the same thing.

**What changed, and whether the gap is as large as expected.** The finetuned model beats both baselines on every measure. As a detector the gap is large: from nothing to finding half the patterns. In money the gap is smaller, 11% cheaper than doing nothing when the model trades on its own. That is the central finding of the project: finding chart patterns turned out to be easier than making money from them.

## Per-class evaluation and failure modes

**Per class on the test days.** Ultralytics reports precision and recall at the confidence that balances the two best.

| Class | Direction | Test boxes | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| Double top | Short | 127 | 0.401 | 0.764 | **0.603** | 0.406 |
| Double bottom | Long | 66 | 0.477 | 0.373 | 0.379 | 0.241 |
| Head and shoulders | Short | 19 | 0.195 | 0.458 | 0.225 | 0.170 |

**The headline number is carried by double tops**: they are 60% of the test boxes and have the best AP50. Head and shoulders drags the average down, but with 19 cases its number cannot be trusted either way.

**Box confusion matrix at the operating threshold of 0.60** (rows are the true class, a match needs an overlap of IoU 0.5):

| True \ Predicted | Double bottom | Double top | Head and shoulders | Missed |
|---|---|---|---|---|
| Double bottom | **10** | 0 | 0 | 56 |
| Double top | 0 | **54** | 0 | 73 |
| Head and shoulders | 0 | 1 | **3** | 15 |
| False alarm on empty chart area | 3 | 19 | 4 | |

The model almost never confuses one pattern with another: 1 box out of 68 it matched went to the wrong class. When it draws a box it is usually right, with a precision of 77% on double bottoms and 73% on double tops. Its weakness is recall. At 0.60 it finds only 15% of double bottoms, 43% of double tops and 16% of head and shoulders.

**Decisions per chart at 0.60** (the model trading on its own, 148 test charts):

| True label \ Decision | Long | Short | No trade |
|---|---|---|---|
| Long (27 charts) | **5** | 1 | 21 |
| Short (70 charts) | 0 | **36** | 34 |
| No trade (51 charts) | 5 | 24 | **22** |

**Where it breaks:**

1. **It leans bearish.** It trades 51% of real short setups but only 19% of real longs. Its worst direction errors are all the same mistake. TSM and IBM on 29 September and QCOM on 28 September each carried a double bottom, and the model's most confident box was a double top (at 0.70, 0.59 and 0.49). At 0.60 only TSM was traded the wrong way, and IBM and QCOM fall in the review band. This follows from training data in which double tops are 56% of the boxes.
2. **Its "spurious" trades are almost all conflicting charts, not invented patterns.** 28 of its 29 trades on "no trade" charts were on charts with both a bullish and a bearish pattern. It traded only one chart that was really empty. Its most confident mistakes are of this kind: IVR at 0.94, AT&T at 0.90 and Block at 0.88 each show a double top and a double bottom, and the model saw only the double top. The model has no notion of conflict. It reports its single most confident box. On conflicting charts that box always lands on a real pattern, but it matches the most recent one only 47% of the time on the test days (43% on validation). On charts whose patterns agree, it points the right way 83% of the time (86% on validation).
3. **Some patterns it does not see at all.** ACN and SQM on 28 September and EUR/USD carried patterns the model scored at 0.10 to 0.12, far below any threshold.

The dashboard's "Where it fails" tab shows these charts, with the true patterns and the model's boxes drawn on them.

## Cost matrix and operating point

**Costs.** As set out in the first section: €1,500 for a missed setup, €1,000 for a spurious trade, €5,000 for a wrong direction trade, €0 for a correct call, and €1.50 plus a 10% chance of error for a chart sent to an analyst.

**Base rate: 57%.** This is the share of training day charts with a clean, single direction pattern (465 of 814). I took it from the training days because they are the largest sample and because doing so leaves the test days untouched. The test days on their own would say 66% and the validation days 44%. Costs are averaged separately over pattern charts and no trade charts and then weighted 57 to 43, so the answer does not depend on each split's own mix.

**The sweep.** Below the threshold the desk does not trade. At or above it, the desk trades the direction of the model's most confident box. I swept the threshold from 0.00 to 1.00 in steps of 0.01 on the validation days. The cost bottoms out at **0.60**, which is the operating point. On the test days the curve is flat between 0.50 and 0.70 (€767,756, €758,685 and €754,084 per 1,000 charts). The test days' own minimum is at 0.71 (€725,086), so the threshold chosen on validation lands within 5% of the best possible on unseen days. Trading every chart (a threshold of 0) costs €929,485, worse than doing nothing, so the model has to be selective. The dashboard's "Cost against threshold" tab shows the curve with the chosen point marked.

**At the operating point, per 1,000 charts:** 241 correct trades, 6 wrong direction trades, 245 spurious trades (almost all on conflicting charts), 323 missed setups and 185 correct passes.

**How I labelled conflicting charts.** I tested two rules, each with its own base rate from the training days.

| | "Most recent pattern wins" | **"Conflicting charts are no trade"** (chosen) |
|---|---|---|
| Base rate from the training days | 77.5% | 57% |
| Real patterns the model trades on its own | 7%, shorts only | 43%, longs and shorts |
| Model alone against doing nothing | 4% cheaper | 11% cheaper |
| Model plus review band against doing nothing | 22% cheaper | 20% cheaper |
| Analysts the review band needs on the test days | 2.54, more than the 2 available | 1.96, within the 2 available |

Under the first rule, a conflicting chart forced the model to pick a side with a €5,000 penalty for picking wrong, so it went almost silent. The second rule is how a trader behaves, lets the model trade both ways, and produces a review load the desk can staff.

**Sensitivity.** Every line below chooses the thresholds and the setup again on the validation days, then reports the test days.

- **Expected move, the assumption that would most change my answer.** I assumed 0.5% because the evidence for chart patterns is thin and I wanted a modest edge. If it falls to 0.2% or less, the trading cost, a missed setup costs nothing and the best decision is to not run the screen at all. From 0.3% upwards, the model with a review band is the best setup.
- **Base rate.** The model trading on its own beats doing nothing only if at least **43%** of charts carry a real pattern. The model with a review band beats doing nothing from **8%** upwards. Below 8%, the thresholds chosen on validation stop trading altogether and every setup collapses to doing nothing.
- **The three error costs and the analyst cost**, each scaled from a quarter to four times my figure: the model with a review band stays the best setup in every case.
- **Analyst accuracy** from 60% to 100%: the band stays the best setup, with a cost between €729,583 and €670,717 per 1,000 charts.

## What a person still has to check

**Two thresholds.** Below **0.49** the desk does not trade and nobody looks. At or above **0.60** it trades automatically. Between the two, the chart goes to an analyst. I chose the pair by sweeping every combination on the validation grid and taking the cheapest one that two analysts can clear.

**The load.** On the test days the band sends 11.8% of charts to review: **59 charts a night** out of 500. At 2 minutes each that is 118 minutes, or **1.96 analysts** giving one evening hour each, against the 2 available. The desk can absorb it, but with no slack: a night with 10% more uncertain charts would overflow. The fix is to cap the queue at the 60 most uncertain charts and to pass the rest as no trade, since at those confidences no trade is the cheaper default.

**Priced against the alternatives** (500 charts a night, cost measured against a perfect trader):

| Setup | Cost per 1,000 charts | Cost per night | Charts reviewed per night | Analysts needed | Possible with 2 analysts |
|---|---|---|---|---|---|
| Do nothing | €855,000 | €427,500 | 0 | 0 | Yes |
| An analyst checks every chart | €130,000 | €65,000 | 500 | 16.7 | No |
| Model alone, threshold 0.60 | €758,685 | €379,343 | 0 | 0 | Yes |
| **Model plus review band, 0.49 to 0.60** | **€685,434** | **€342,717** | **59** | **1.96** | **Yes** |

Doing nothing gives up the whole edge on offer: 285 real patterns a night at €1,500 each, or €427,500. The model with a review band recovers about **€84,800 a night, roughly a fifth of that edge**, for two evening hours of analyst time. An analyst checking every chart would be the cheapest setup of all, since a person who is right 9 times in 10 beats this model by a wide margin. But it needs almost 17 people, and the desk has 2.

**Staffing is the strongest lever.** With 1 analyst the cost is €738,336 per 1,000 charts, with 2 it is €685,434, with 3 it is €624,608 and with 5 it is €537,818. A third analyst hour at €45 would cut the loss by about €30,400 a night.

**Verdict.** I would not let the model trade on its own: it saves only 11% against doing nothing and leans short. I would ship it as a triage screen with the review band and the two analysts, after a paper trading pilot that measures the real edge. If the pilot confirms an edge above the trading cost, the cheapest improvement is a third evening hour of analyst time, not a better model.

**The dashboard** (`uv run python main.py app`) loads the committed weights and saved results and never trains. Moving the two thresholds updates the confusion matrix, the expected cost per 1,000 charts and the share sent to review together, and a base rate slider shows how the answer moves. Other tabs show the cost curve with the chosen threshold, the per class results, the failure cases, the sensitivity analysis, and a chart upload that shows what the model would do.

## Limitations & next steps

1. **The edge is assumed, not measured.** Next step: download the next day's open to close return for every test chart from Yahoo Finance and measure the hit rate and average move of each pattern. That replaces the 0.5% assumption with evidence.
2. **The euro figures ignore the size of the book.** At a 57% base rate a perfect trader would open about 285 positions a night, €142m of exposure on a €100m book, and the review band setup opens about 246. A real desk would cap the number of trades and take the strongest signals. The comparison between setups holds because every setup is costed per chart, but the euro totals are overstated. Next step: add a nightly position cap to the cost model.
3. **The model has no notion of conflict.** Next step: change the decision rule so the desk does not trade when boxes pointing opposite ways both clear the lower threshold. On the test days that would target most of the 245 spurious trades per 1,000 charts.
4. **It leans bearish, and head and shoulders is too thin to judge.** Next step: add the same uploader's 5 minute dataset (794 charts) as extra training data or as a second test set, and weight the classes so long setups count as much as short ones.
5. **One regime, the same 74 names, two test days.** Next step: test on other periods and on stocks outside the training set, with a walk forward evaluation that retrains on a rolling window, before any capital is at risk.
6. **The labels come from one person and patterns are subjective.** Next step: have a second person label the test charts and measure how often the two agree. That also gives a measured analyst accuracy in place of my assumed 90%.
7. **Small patterns lose detail at 640 pixels, and mosaic augmentation stitches four charts into one**, creating charts where time jumps between stocks. Next step: train at 960 or 1,024 pixels and with mosaic switched off, and compare the runs on validation only.
8. **The review queue has no slack.** Next step: the queue cap described above, and a weekly check that the review share has not drifted.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment.

**What Claude did:**

1. Shortlisted chart pattern datasets on Roboflow Universe and compared them against the rubric. I chose `stocks-3patterns-2m-sept2023`.
2. Wrote all the code (`train.py`, `app.py`, `costing.py` and `config.py`): the download, the split by date, the training call, the baselines, the evaluation, the threshold sweeps, the sensitivity analysis and the dashboard. It tested the pipeline on synthetic data before my real run and checked my committed weights and dashboard afterwards.
3. Proposed design options with their trade offs, including the split by date, choosing thresholds on the validation days, switching off flips and the two rules for conflicting charts. It ran the comparison of the two rules on my results.
4. Proposed starting figures for the cost model from my inputs and from published references, and explained how a fund would set them.
5. Drafted the text of this report from my decisions and the pipeline's outputs, at my request.

**What I decided:**

1. The business problem: an end of day screen at a long/short fund, trading stocks only.
2. The dataset, out of the shortlist.
3. The desk inputs: two analysts at €60,000 a year, 20 bps of trading costs, and the view that a trade on noise only loses its costs.
4. Adopting the position size, expected move and review assumptions after discussing how a fund would set them.
5. Labelling conflicting charts as no trade and using the training days' 57% base rate, after seeing the comparison.
6. The verdict to deploy the model as a triage screen after a pilot.

I have reviewed every section and every figure in this report and can explain each decision without the code in front of me.
