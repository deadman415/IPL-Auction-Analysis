# Power BI Setup Guide — BA Project
# ================================================
# Follow these steps exactly after running the Python pipeline

## STEP 1 — Import Data into Power BI
# File → Get Data → Text/CSV
# Import these files (all in the /models/ folder):
#   - processed_data.csv       (main dataset — 626 rows)
#   - franchise_efficiency.csv (franchise summary)
#   - top_value_players.csv    (top 20 value picks)

## STEP 2 — Data Types (set these in Power Query Editor)
# processed_data:
#   year             → Whole Number
#   sold_price_cr    → Decimal Number
#   base_price_cr    → Decimal Number
#   value_index      → Decimal Number
#   performance_score→ Decimal Number
#   is_sold          → Whole Number

## STEP 3 — DAX Measures (add these in Report View → New Measure)

# ── Basic KPIs ──────────────────────────────────────────────────────────────

Total Players Auctioned = COUNTROWS(processed_data)

Total Players Sold = CALCULATE(COUNTROWS(processed_data), processed_data[is_sold] = 1)

Avg Sale Price (Cr) = 
CALCULATE(
    AVERAGE(processed_data[sold_price_cr]),
    processed_data[is_sold] = 1
)

Max Sale Price (Cr) = 
CALCULATE(
    MAX(processed_data[sold_price_cr]),
    processed_data[is_sold] = 1
)

Unsold Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(processed_data), processed_data[is_sold] = 0),
    COUNTROWS(processed_data),
    0
) * 100

# ── Advanced Measures ────────────────────────────────────────────────────────

Price Growth YoY % = 
VAR CurrentYear = MAX(processed_data[year])
VAR CurrentAvg = CALCULATE(AVERAGE(processed_data[sold_price_cr]),
                            processed_data[year] = CurrentYear)
VAR PrevAvg = CALCULATE(AVERAGE(processed_data[sold_price_cr]),
                         processed_data[year] = CurrentYear - 1)
RETURN DIVIDE(CurrentAvg - PrevAvg, PrevAvg, 0) * 100

Avg Value Index = AVERAGE(processed_data[value_index])

Top Value Player = 
CALCULATE(
    FIRSTNONBLANK(processed_data[player_name], 1),
    TOPN(1, processed_data, processed_data[value_index], DESC)
)

Franchise Efficiency Rank = 
RANKX(
    ALL(processed_data[franchise]),
    CALCULATE(AVERAGE(processed_data[value_index])),
    ,
    DESC
)

Overseas Premium % = 
VAR OverseaAvg = CALCULATE(AVERAGE(processed_data[sold_price_cr]),
                             processed_data[nationality] <> "Indian")
VAR IndianAvg  = CALCULATE(AVERAGE(processed_data[sold_price_cr]),
                             processed_data[nationality] = "Indian")
RETURN DIVIDE(OverseaAvg - IndianAvg, IndianAvg, 0) * 100

## STEP 4 — Page Layout (create 4 pages)

# PAGE 1 — Executive Overview
# ─────────────────────────────
# Top row:  4 KPI cards using measures above
#   → Total Players Auctioned | Total Players Sold | Avg Sale Price | Max Sale Price
#
# Middle row (2 charts):
#   → Line chart: X=year, Y=Avg Sale Price (Cr) [add secondary line for Median]
#   → Donut chart: Role breakdown of sold players (Legend=role, Values=count)
#
# Bottom row (2 charts):
#   → Stacked bar: X=year, Y=sold_price_cr sum, Legend=franchise (top 5)
#   → Map or treemap: franchise → total spend
#
# Slicers (right panel): Year (range), Role (checkboxes), Nationality (dropdown)

# PAGE 2 — Player Deep Dive
# ─────────────────────────────
# Top: Scatter chart
#   → X=performance_score, Y=sold_price_cr, Size=ipl_experience_years
#   → Color=role, Tooltip=player_name, year, franchise
#
# Middle: Table visual
#   → Columns: player_name, role, nationality, year, batting_avg, batting_sr,
#              wickets, economy_rate, sold_price_cr, value_index
#   → Sort by value_index descending by default
#
# Bottom: Bar chart — Top 10 players by sold_price_cr for selected filters

# PAGE 3 — Value Intelligence
# ─────────────────────────────
# Bubble chart:
#   → X=sold_price_cr, Y=performance_score, Bubble size=value_index
#   → Color=role
#   → Add reference lines at avg price and avg performance (quadrant view)
#   → TOP RIGHT quadrant = Stars | BOTTOM LEFT = Bargains
#
# Bar chart: Top 15 players by value_index, colored by role
#
# Card: "Best Value Player This Year" using Top Value Player measure
#
# Table: top_value_players.csv import — show all 20

# PAGE 4 — Franchise Strategy
# ─────────────────────────────
# Scatter: X=total_spend, Y=avg_performance (from franchise_efficiency)
#   → Size=players_bought, Color=avg_value_index, Labels=franchise
#
# Bar: Franchise Efficiency Rank (avg_value_index, sorted descending)
#   → Color-code: green for top 5, red for bottom 5
#
# Area chart: yearly total spend by franchise
#
# Table: franchise_efficiency.csv — all columns, conditional formatting on avg_value_index

## STEP 5 — Formatting Tips for High Marks
# Theme colors:
#   Primary:   #378ADD  (blue)
#   Secondary: #1D9E75  (green)
#   Accent:    #EF9F27  (amber)
#   Danger:    #D85A30  (coral)
#
# - Use consistent font: Segoe UI
# - All chart titles: 14pt, bold
# - Enable gridlines: light gray (#f0ede8)
# - Add data labels to bar/bar-horizontal charts
# - Use conditional formatting on value_index column in tables:
#     → Above average = green background
#     → Below average = red background
# - Add tooltips: hover over any player to see their full profile

## STEP 6 — Publish & Share
# File → Publish to Power BI Service (free account)
# Get shareable link → include in report and PPT
