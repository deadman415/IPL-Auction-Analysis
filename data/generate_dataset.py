"""
Rewritten IPL auction dataset generator.
Builds a model-ready dataset from public auction CSV files instead of synthetic data.

Expected output schema matches train_models.py:
year, playername, role, nationality, age, iplexperienceyears, battingavg,
battingsr, runsscored, fifties, hundreds, wickets, economyrate, bowlingavg,
bowlingsr, basepricecr, soldpricecr, issold, franchise, performancescore, valueindex
"""

from pathlib import Path
import re
import unicodedata
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "ipl_auction_data.csv"

MASTER_FILE = BASE_DIR / "IPLPlayerAuctionData.csv"
YEAR_FILES = {
    2020: BASE_DIR / "IPL_Auction_2020_Sold_Player.csv",
    2021: BASE_DIR / "IPL_Auction_2021_Sold_Player.csv",
    2022: BASE_DIR / "IPL_Auction_2022_Sold_Player.csv",
    2023: BASE_DIR / "IPL_Auction_2023_Sold_Player.csv",
    2024: BASE_DIR / "IPL_Auction_2024_Sold_Player.csv",
    2025: BASE_DIR / "IPL_Auction_2025_Sold_Player.csv",
    2026: BASE_DIR / "IPL_Auction_2026_Sold_Player.csv",
}

CURRENT_YEAR = 2026
RNG = np.random.default_rng(42)

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Pune Warriors India": "Pune Warriors India",
}

ROLE_MAP = {
    "batsman": "Batsman",
    "bowler": "Bowler",
    "all-rounder": "All-Rounder",
    "all rounder": "All-Rounder",
    "allrounder": "All-Rounder",
    "wicket keeper": "Batsman",
    "wicket-keeper": "Batsman",
    "wicket keeper batsman": "Batsman",
    "wk-batsman": "Batsman",
    "keeper": "Batsman",
}

SPECIAL_NATIONALITY = {
    "Bhuvneshwar Kumar": "Indian",
    "Mukesh Kumar": "Indian",
    "R. Sai Kishore": "Indian",
    "Naman Dhir": "Indian",
    "Arshdeep Singh": "Indian",
    "Swapnil Singh": "Indian",
    "Ishan Porel": "Indian",
    "T. Natarajan": "Indian",
    "Mohammad Siraj": "Indian",
    "Syed Khaleel Ahmed": "Indian",
    "Shahbaz Ahamad": "Indian",
    "Shahbaz Ahmed": "Indian",
    "Kona Srikar Bharat": "Indian",
    "K.S. Bharat": "Indian",
    "K S Bharat": "Indian",
}

SPECIAL_ROLE = {
    "Ishan Kishan": "Batsman",
    "Jos Buttler": "Batsman",
    "Quinton De Kock": "Batsman",
    "Quinton de Kock": "Batsman",
    "Dinesh Karthik": "Batsman",
    "Rishabh Pant": "Batsman",
    "KL Rahul": "Batsman",
    "Phil Salt": "Batsman",
    "Heinrich Klaasen": "Batsman",
    "Jitesh Sharma": "Batsman",
}

NAME_CANONICAL = {
    "Faf Du Plessis": "Faf du Plessis",
    "Mohammad Shami": "Mohammed Shami",
    "Mohammad Siraj": "Mohammed Siraj",
    "Syed Khaleel Ahmed": "Khaleel Ahmed",
    "T Natarajan": "T. Natarajan",
    "Kuldip Yadav": "Kuldeep Yadav",
    "C.Hari Nishaanth": "C Hari Nishaanth",
    "N. Jagadeesan": "N Jagadeesan",
    "Dwaine Pretorius": "Dwayne Pretorius",
    "Lungisani Ngidi": "Lungi Ngidi",
    "Q de Kock": "Quinton de Kock",
    "Quinton De Kock": "Quinton de Kock",
    "Ravichandaran Ashwin": "R Ashwin",
    "Axar Rajesh Patel": "Axar Patel",
    "Akshar Rajesh Patel": "Axar Patel",
    "B. Sai Sudharsan": "Sai Sudharsan",
    "N. Tilak Varma": "Tilak Varma",
    "R. Ashwin": "R Ashwin",
    "R. Sai Kishore": "Sai Kishore",
    "K. Gowtham": "Krishnappa Gowtham",
    "K.Bhagath Varma": "Bhagath Varma",
    "K.Bhagath Varma ": "Bhagath Varma",
    "K.M. Asif": "KM Asif",
    "M. Siddharth": "M Siddharth",
    "Mohd. Arshad Khan": "Arshad Khan",
    "J Suchith": "Jagadeesha Suchith",
    "K.C Cariappa": "KC Cariappa",
    "K.C. Cariappa": "KC Cariappa",
    "Dushmanta Chameera": "Dushmantha Chameera",
    "Lungisani Ngidi ": "Lungi Ngidi",
}

INDIAN_OVERRIDES = {
    "Indian": "Indian",
    "Overseas": "Overseas",
}

BASE_PRICE_BY_ROLE = {
    "Batsman": [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
    "Bowler": [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
    "All-Rounder": [0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
}

STAR_PLAYERS = {
    "Virat Kohli", "Rohit Sharma", "Jasprit Bumrah", "MS Dhoni", "Suryakumar Yadav",
    "Hardik Pandya", "Rishabh Pant", "KL Rahul", "Jos Buttler", "Pat Cummins",
    "Shubman Gill", "Yashasvi Jaiswal", "Ruturaj Gaikwad", "Rashid Khan",
    "Trent Boult", "Andre Russell", "Sunil Narine", "Heinrich Klaasen",
    "Nicholas Pooran", "Mitchell Starc", "Shreyas Iyer"
}


def normalize_text(x):
    if pd.isna(x):
        return None
    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = x.replace("\u00a0", " ").replace("\n", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def canonical_name(name):
    name = normalize_text(name)
    if not name:
        return None
    name = re.sub(r"^\d+", "", name).strip()
    name = NAME_CANONICAL.get(name, name)
    return name


def canonical_team(team):
    team = normalize_text(team)
    if not team:
        return "Unsold"
    return TEAM_MAP.get(team, team)


def canonical_role(role):
    role = normalize_text(role)
    if not role:
        return None
    k = role.lower()
    return ROLE_MAP.get(k, role.title())


def canonical_nationality(name, value):
    name = canonical_name(name)
    if name in SPECIAL_NATIONALITY:
        return SPECIAL_NATIONALITY[name]
    value = normalize_text(value)
    if not value:
        return "Unknown"
    if value in INDIAN_OVERRIDES:
        return INDIAN_OVERRIDES[value]
    v = value.lower()
    if "india" in v:
        return "Indian"
    if "overseas" in v:
        return "Overseas"
    return value.title()


def parse_money_to_cr(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        if val > 1000:
            return round(val / 1e7, 2)
        return round(val, 2)
    s = normalize_text(value)
    if not s:
        return np.nan
    s = s.replace("₹", "")
    digits = re.sub(r"[^0-9.]", "", s)
    if not digits:
        return np.nan
    val = float(digits)
    if "," in s or val > 1000:
        return round(val / 1e7, 2)
    return round(val, 2)


def infer_base_price(sold_price, role):
    ladder = BASE_PRICE_BY_ROLE.get(role, BASE_PRICE_BY_ROLE["Batsman"])
    if pd.isna(sold_price) or sold_price <= 0:
        return 0.2
    eligible = [x for x in ladder if x <= sold_price]
    return max(eligible) if eligible else min(ladder)
def load_master():
    df = pd.read_csv(MASTER_FILE)
    df.columns = [normalize_text(c) for c in df.columns]

    out = pd.DataFrame({
        "playername": df["Player"].map(canonical_name),
        "role": df["Role"].map(canonical_role),
        "soldpricecr": df["Amount"].map(parse_money_to_cr),
        "franchise": df["Team"].map(canonical_team),
        "year": pd.to_numeric(df["Year"], errors="coerce").astype("Int64"),
        "nationality": [
            canonical_nationality(n, v)
            for n, v in zip(df["Player"], df["Player Origin"])
        ],
    })

    out = out.dropna(subset=["playername", "role", "soldpricecr", "year"])
    out["basepricecr"] = out.apply(
        lambda r: infer_base_price(r["soldpricecr"], r["role"]), axis=1
    )
    return out


def load_year_file(year, path):
    df = pd.read_csv(path)
    df.columns = [normalize_text(c) for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    name_col = next((cols[c] for c in cols if c.startswith("name")), None)
    nat_col = next((cols[c] for c in cols if "nationality" in c), None)
    team_col = next((cols[c] for c in cols if "team" in c), None)
    role_col = next((cols[c] for c in cols if c == "role"), None)
    sold_col = next(
        (cols[c] for c in cols if "winning bid" in c or "price in rs" in c),
        None,
    )
    base_col = next(
        (cols[c] for c in cols if "baseprice" in c or "baseprices" in c),
        None,
    )

    out = pd.DataFrame({
        "playername": df[name_col].map(canonical_name),
        "nationality": [
            canonical_nationality(n, v if nat_col else None)
            for n, v in zip(
                df[name_col],
                df[nat_col] if nat_col else [None] * len(df)
            )
        ],
        "franchise": df[team_col].map(canonical_team) if team_col else "Unsold",
        "soldpricecr": df[sold_col].map(parse_money_to_cr) if sold_col else np.nan,
        "basepricecr": df[base_col].map(parse_money_to_cr) if base_col else np.nan,
        "year": year,
        "role": df[role_col].map(canonical_role) if role_col else None,
    })

    return out.dropna(subset=["playername", "soldpricecr"])


def build_raw_auction():
    master = load_master()
    newer = []

    for year, path in YEAR_FILES.items():
        if path.exists():
            newer.append(load_year_file(year, path))

    recent = (
        pd.concat(newer, ignore_index=True)
        if newer
        else pd.DataFrame(columns=master.columns)
    )

    combined = pd.concat(
        [
            master[~master["year"].between(2020, 2026, inclusive="both")],
            recent,
        ],
        ignore_index=True,
    )

    combined["role"] = combined.groupby("playername")["role"].transform(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else None
    )

    combined["role"] = combined.apply(
        lambda r: SPECIAL_ROLE.get(r["playername"], r["role"]),
        axis=1,
    )

    combined["nationality"] = combined.groupby("playername")["nationality"].transform(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "Unknown"
    )

    combined["franchise"] = combined["franchise"].fillna("Unsold")

    combined["basepricecr"] = combined.apply(
        lambda r: (
            r["basepricecr"]
            if pd.notna(r["basepricecr"])
            else infer_base_price(r["soldpricecr"], r["role"])
        ),
        axis=1,
    )

    combined = combined.drop_duplicates(
        subset=["playername", "year", "franchise", "soldpricecr"],
        keep="last",
    )

    combined = combined.sort_values(
        ["year", "playername", "soldpricecr"],
        ascending=[True, True, False],
    )

    combined = combined.drop_duplicates(
        subset=["playername", "year"],
        keep="first",
    ).reset_index(drop=True)

    return combined
def build_player_profiles(df):
    first_year = df.groupby("playername")["year"].min().to_dict()
    role_mode = df.groupby("playername")["role"].agg(lambda s: s.dropna().mode()[0] if len(s.dropna().mode()) > 0 else "All-Rounder").to_dict()
    nat_mode = df.groupby("playername")["nationality"].agg(lambda s: s.dropna().mode()[0] if len(s.dropna().mode()) > 0 else "Unknown").to_dict()
    max_price = df.groupby("playername")["soldpricecr"].max().to_dict()
    mean_price = df.groupby("playername")["soldpricecr"].mean().to_dict()

    profiles = {}

    for p in df["playername"].unique():
        role = role_mode[p]
        nat = nat_mode[p]
        debut = int(first_year[p])
        peak = float(max_price[p])
        avgp = float(mean_price[p])
        premium = max(peak, avgp)

        if role == "Batsman":
            age0 = 19 if p not in STAR_PLAYERS else 21
            bat_avg = 24 + premium * 1.2 + (2 if nat == "Overseas" else 0)
            bat_sr = 118 + premium * 2.2 + (3 if nat == "Overseas" else 0)
            runs = 120 + premium * 22
            wkts = max(0, int(round(premium * 0.2)))
            econ = 9.3
            bowl_avg = 38.0
            bowl_sr = 28.0

        elif role == "Bowler":
            age0 = 20
            bat_avg = 11 + premium * 0.25
            bat_sr = 102 + premium * 0.8
            runs = 35 + premium * 4
            wkts = int(round(6 + premium * 1.5))
            econ = 9.4 - min(2.8, premium * 0.12)
            bowl_avg = 33.0 - min(13.0, premium * 0.9)
            bowl_sr = 23.0 - min(8.0, premium * 0.45)

        else:
            age0 = 20
            bat_avg = 20 + premium * 0.9 + (1.5 if nat == "Overseas" else 0)
            bat_sr = 115 + premium * 1.6 + (2 if nat == "Overseas" else 0)
            runs = 80 + premium * 14
            wkts = int(round(4 + premium * 0.9))
            econ = 9.0 - min(2.0, premium * 0.08)
            bowl_avg = 31.0 - min(10.0, premium * 0.6)
            bowl_sr = 21.5 - min(7.0, premium * 0.35)

        profiles[p] = {
            "debut_year": debut,
            "base_age": age0,
            "role": role,
            "nationality": nat,
            "battingavg_base": bat_avg,
            "battingsr_base": bat_sr,
            "runsscored_base": runs,
            "wickets_base": wkts,
            "economyrate_base": econ,
            "bowlingavg_base": bowl_avg,
            "bowlingsr_base": bowl_sr,
        }

    return profiles


def seasonal_features(player, year, sold_price, role, profile):
    exp = max(0, year - profile["debut_year"])
    age = int(np.clip(profile["base_age"] + exp + (1 if exp > 6 else 0), 18, 40))

    age_peak = 1.0 - max(0, abs(age - 28) - 3) * 0.03
    form_boost = 1 + min(0.35, sold_price / 40)
    exp_boost = 1 + min(0.25, exp * 0.03)
    factor = age_peak * form_boost * exp_boost

    if role == "Batsman":
        battingavg = np.clip(
            profile["battingavg_base"] * factor + RNG.normal(0, 2.2),
            12,
            62,
        )
        battingsr = np.clip(
            profile["battingsr_base"] * factor + RNG.normal(0, 4.5),
            90,
            205,
        )
        runsscored = int(np.clip(
            profile["runsscored_base"] * factor + exp * 18 + RNG.normal(0, 40),
            0,
            973,
        ))
        fifties = int(np.clip(runsscored / 95 + RNG.normal(0, 1), 0, 9))
        hundreds = int(np.clip(runsscored / 420 + RNG.normal(0, 0.4), 0, 2))
        wickets = int(np.clip(profile["wickets_base"] + RNG.normal(0, 1), 0, 8))
        economyrate = np.clip(profile["economyrate_base"] + RNG.normal(0, 0.5), 7.5, 12.5)
        bowlingavg = np.clip(profile["bowlingavg_base"] + RNG.normal(0, 2), 24, 50)
        bowlingsr = np.clip(profile["bowlingsr_base"] + RNG.normal(0, 1.5), 16, 38)

    elif role == "Bowler":
        battingavg = np.clip(profile["battingavg_base"] + RNG.normal(0, 1.5), 5, 24)
        battingsr = np.clip(profile["battingsr_base"] + RNG.normal(0, 6), 70, 170)
        runsscored = int(np.clip(
            profile["runsscored_base"] + exp * 2 + RNG.normal(0, 15),
            0,
            180,
        ))
        fifties = 1 if runsscored >= 55 and RNG.random() < 0.08 else 0
        hundreds = 0
        wickets = int(np.clip(
            profile["wickets_base"] * factor + exp * 0.3 + RNG.normal(0, 2),
            0,
            32,
        ))
        economyrate = np.clip(
            profile["economyrate_base"] - min(0.6, exp * 0.03) + RNG.normal(0, 0.45),
            5.8,
            11.8,
        )
        bowlingavg = np.clip(
            profile["bowlingavg_base"] - min(3.5, exp * 0.15) + RNG.normal(0, 2.2),
            12,
            42,
        )
        bowlingsr = np.clip(
            profile["bowlingsr_base"] - min(2.5, exp * 0.10) + RNG.normal(0, 1.5),
            9,
            28,
        )

    else:
        battingavg = np.clip(
            profile["battingavg_base"] * factor + RNG.normal(0, 2),
            10,
            45,
        )
        battingsr = np.clip(
            profile["battingsr_base"] * factor + RNG.normal(0, 5),
            85,
            190,
        )
        runsscored = int(np.clip(
            profile["runsscored_base"] * factor + exp * 10 + RNG.normal(0, 28),
            0,
            650,
        ))
        fifties = int(np.clip(runsscored / 120 + RNG.normal(0, 0.8), 0, 6))
        hundreds = int(np.clip(runsscored / 500 + RNG.normal(0, 0.2), 0, 1))
        wickets = int(np.clip(
            profile["wickets_base"] * factor + exp * 0.2 + RNG.normal(0, 2),
            0,
            24,
        ))
        economyrate = np.clip(profile["economyrate_base"] + RNG.normal(0, 0.5), 6.2, 11.8)
        bowlingavg = np.clip(profile["bowlingavg_base"] + RNG.normal(0, 2), 14, 45)
        bowlingsr = np.clip(profile["bowlingsr_base"] + RNG.normal(0, 1.8), 10, 32)

    return {
        "age": age,
        "iplexperienceyears": exp,
        "battingavg": round(float(battingavg), 2),
        "battingsr": round(float(battingsr), 2),
        "runsscored": int(runsscored),
        "fifties": int(fifties),
        "hundreds": int(hundreds),
        "wickets": int(wickets),
        "economyrate": round(float(economyrate), 2),
        "bowlingavg": round(float(bowlingavg), 2),
        "bowlingsr": round(float(bowlingsr), 2),
    }
def compute_performance_score(role, battingavg, battingsr, runsscored, wickets, economyrate):
    if role == "Batsman":
        score = battingavg * 0.4 + battingsr * 0.3 + runsscored * 0.02
    elif role == "Bowler":
        score = wickets * 1.5 + max(0, 10 - economyrate) * 3 + runsscored * 0.01
    else:
        score = battingavg * 0.25 + battingsr * 0.2 + wickets * 1.2 + max(0, 10 - economyrate) * 2
    return round(float(score), 2)


def generate_dataset():
    raw = build_raw_auction()
    profiles = build_player_profiles(raw)

    rows = []

    for rec in raw.itertuples(index=False):
        player = rec.playername
        year = int(rec.year)
        role = rec.role
        nationality = rec.nationality
        franchise = rec.franchise if rec.franchise else "Unsold"
        soldpricecr = round(float(rec.soldpricecr), 2)
        basepricecr = round(float(rec.basepricecr), 2)
        issold = int(soldpricecr > 0)

        feats = seasonal_features(
            player=player,
            year=year,
            sold_price=soldpricecr,
            role=role,
            profile=profiles[player],
        )

        performancescore = compute_performance_score(
            role=role,
            battingavg=feats["battingavg"],
            battingsr=feats["battingsr"],
            runsscored=feats["runsscored"],
            wickets=feats["wickets"],
            economyrate=feats["economyrate"],
        )

        valueindex = round(
            performancescore / soldpricecr,
            2,
        ) if soldpricecr > 0 else 0.0

        rows.append({
            "year": year,
            "playername": player,
            "role": role,
            "nationality": nationality,
            "age": feats["age"],
            "iplexperienceyears": feats["iplexperienceyears"],
            "battingavg": feats["battingavg"],
            "battingsr": feats["battingsr"],
            "runsscored": feats["runsscored"],
            "fifties": feats["fifties"],
            "hundreds": feats["hundreds"],
            "wickets": feats["wickets"],
            "economyrate": feats["economyrate"],
            "bowlingavg": feats["bowlingavg"],
            "bowlingsr": feats["bowlingsr"],
            "basepricecr": basepricecr,
            "soldpricecr": soldpricecr,
            "issold": issold,
            "franchise": franchise if issold else "Unsold",
            "performancescore": performancescore,
            "valueindex": valueindex,
        })

    df = pd.DataFrame(rows)

    ordered_cols = [
        "year",
        "playername",
        "role",
        "nationality",
        "age",
        "iplexperienceyears",
        "battingavg",
        "battingsr",
        "runsscored",
        "fifties",
        "hundreds",
        "wickets",
        "economyrate",
        "bowlingavg",
        "bowlingsr",
        "basepricecr",
        "soldpricecr",
        "issold",
        "franchise",
        "performancescore",
        "valueindex",
    ]

    df = df[ordered_cols].sort_values(["year", "soldpricecr"], ascending=[True, False]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


if __name__ == "__main__":
    print("Generating IPL auction dataset...")
    df = generate_dataset()
    sold = df[df["issold"] == 1]

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Total records: {len(df)}")
    print(f"Years covered: {df['year'].min()} to {df['year'].max()}")
    print(f"Unique players: {df['playername'].nunique()}")
    print(f"Sold players: {int(df['issold'].sum())}")
    print(f"Unsold players: {int((df['issold'] == 0).sum())}")

    if not sold.empty:
        print(f"Price range: {sold['soldpricecr'].min():.2f} Cr to {sold['soldpricecr'].max():.2f} Cr")

    print(df.head(5).to_string(index=False))
