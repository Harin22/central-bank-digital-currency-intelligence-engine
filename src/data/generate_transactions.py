import os
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_TRANSACTIONS = 100_000
N_WALLETS = 10_000
N_MERCHANTS = 2_000

WARMUP_DAYS = 30
WARMUP_TRANSACTIONS = 80_000

START_DATE = "2026-01-01"
END_DATE = "2026-07-01"

OUTPUT_PATH = "data/raw/normal_transactions.csv"

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# POPULATION
# ============================================================

COUNTRIES = [
    "DE", "FR", "IT", "ES", "NL",
    "BE", "AT", "PT", "IE", "FI",
    "GR", "LU", "SK", "SI", "EE",
    "LV", "LT", "CY", "MT", "HR"
]

PSPS = [
    "PSP01", "PSP02", "PSP03",
    "PSP04", "PSP05"
]

MERCHANT_CATEGORIES = [
    "grocery",
    "transport",
    "restaurant",
    "retail",
    "utilities",
    "entertainment",
    "healthcare",
    "travel",
    "electronics",
    "other"
]

TRANSACTION_TYPES = [
    "p2p_transfer",
    "purchase"
]

PAYMENT_MODES = [
    "online",
    "offline"
]


# ============================================================
# WALLET ACTIVITY PROFILES
# ============================================================

PROFILES = {
    "low": {
        "probability": 0.20,
        "activity_weight": 0.5
    },
    "normal": {
        "probability": 0.50,
        "activity_weight": 1.0
    },
    "active": {
        "probability": 0.20,
        "activity_weight": 2.5
    },
    "very_active": {
        "probability": 0.10,
        "activity_weight": 5.0
    }
}


# ============================================================
# HELPERS
# ============================================================

def generate_amount(transaction_type):

    if transaction_type == "purchase":

        amount = np.random.lognormal(
            mean=3.9,
            sigma=0.75
        )

        amount = np.clip(
            amount,
            2,
            1500
        )

    else:

        amount = np.random.lognormal(
            mean=4.4,
            sigma=0.9
        )

        amount = np.clip(
            amount,
            5,
            5000
        )

    return round(float(amount), 2)


def get_transaction_type(profile):

    if profile == "very_active":
        weights = [0.65, 0.35]

    elif profile == "active":
        weights = [0.60, 0.40]

    else:
        weights = [0.55, 0.45]

    return random.choices(
        TRANSACTION_TYPES,
        weights=weights,
        k=1
    )[0]


def get_payment_mode(transaction_type):

    if transaction_type == "purchase":

        return random.choices(
            PAYMENT_MODES,
            weights=[0.75, 0.25],
            k=1
        )[0]

    return random.choices(
        PAYMENT_MODES,
        weights=[0.65, 0.35],
        k=1
    )[0]


def generate_timestamp(start, end):

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    seconds = int(
        (end_ts - start_ts).total_seconds()
    )

    random_seconds = random.randint(
        0,
        seconds
    )

    return (
        start_ts
        + pd.Timedelta(
            seconds=random_seconds
        )
    )


def generate_device_id():

    return f"DEV{random.randint(100000, 999999)}"


# ============================================================
# CREATE WALLETS
# ============================================================

print("Creating wallet population...")

wallet_ids = [
    f"W{i:05d}"
    for i in range(1, N_WALLETS + 1)
]

profile_names = list(
    PROFILES.keys()
)

profile_weights = [
    PROFILES[p]["probability"]
    for p in profile_names
]

wallet_profiles = np.random.choice(
    profile_names,
    size=N_WALLETS,
    p=profile_weights
)

wallets = {}

for wallet_id, profile in zip(
    wallet_ids,
    wallet_profiles
):

    home_country = random.choice(
        COUNTRIES
    )

    wallets[wallet_id] = {

        "profile": profile,

        "home_country": home_country,

        "balance": round(
            float(
                np.random.lognormal(
                    mean=9.0,
                    sigma=0.7
                )
            ),
            2
        ),

        "device_id": generate_device_id(),

        "last_location": home_country
    }

print(
    f"Created {N_WALLETS:,} persistent wallets."
)


# ============================================================
# CREATE MERCHANTS
# ============================================================

print("Creating merchant population...")

merchant_ids = [
    f"M{i:05d}"
    for i in range(1, N_MERCHANTS + 1)
]

merchants = {}

for merchant_id in merchant_ids:

    merchants[merchant_id] = {

        "category": random.choice(
            MERCHANT_CATEGORIES
        ),

        "country": random.choice(
            COUNTRIES
        )
    }

print(
    f"Created {N_MERCHANTS:,} merchants."
)


# ============================================================
# WALLET SAMPLING
# ============================================================

wallet_activity_weights = np.array([

    PROFILES[
        wallets[w]["profile"]
    ]["activity_weight"]

    for w in wallet_ids
])

wallet_activity_weights /= (
    wallet_activity_weights.sum()
)


def choose_sender():

    return np.random.choice(
        wallet_ids,
        p=wallet_activity_weights
    )


# ============================================================
# TRANSACTION HISTORY
#
# IMPORTANT:
# Sender history is retained for 7 DAYS.
# This allows:
#   1h  -> 1 hour window
#   24h -> 24 hour window
#   7d  -> 7 day window
#
# Receiver history only needs 24 hours.
# ============================================================

sender_history = defaultdict(deque)
receiver_history = defaultdict(deque)


def cleanup_sender_history(
    wallet_id,
    current_time
):

    history = sender_history[wallet_id]

    seven_days_ago = (
        current_time
        - pd.Timedelta(days=7)
    )

    while (
        history
        and history[0][0] < seven_days_ago
    ):

        history.popleft()


def cleanup_receiver_history(
    wallet_id,
    current_time
):

    history = receiver_history[wallet_id]

    one_day_ago = (
        current_time
        - pd.Timedelta(days=1)
    )

    while (
        history
        and history[0][0] < one_day_ago
    ):

        history.popleft()


# ============================================================
# BEHAVIOURAL FEATURES
# ============================================================

def calculate_features(
    sender_id,
    receiver_id,
    timestamp,
    amount,
    receiver_is_wallet
):

    # Keep sender history for 7 days
    cleanup_sender_history(
        sender_id,
        timestamp
    )

    if receiver_is_wallet:

        cleanup_receiver_history(
            receiver_id,
            timestamp
        )

    one_hour_ago = (
        timestamp
        - pd.Timedelta(hours=1)
    )

    one_day_ago = (
        timestamp
        - pd.Timedelta(days=1)
    )

    seven_days_ago = (
        timestamp
        - pd.Timedelta(days=7)
    )

    # --------------------------------------------------------
    # Sender velocity / transaction counts
    # --------------------------------------------------------

    sender_transactions_1h = 0
    sender_transactions_24h = 0

    for ts, _ in sender_history[sender_id]:

        if ts >= one_hour_ago:
            sender_transactions_1h += 1

        if ts >= one_day_ago:
            sender_transactions_24h += 1

    # --------------------------------------------------------
    # Receiver velocity
    # --------------------------------------------------------

    receiver_transactions_1h = 0

    if receiver_is_wallet:

        for ts, _ in receiver_history[receiver_id]:

            if ts >= one_hour_ago:
                receiver_transactions_1h += 1

    # --------------------------------------------------------
    # TRUE 7-DAY AMOUNT HISTORY
    # --------------------------------------------------------

    recent_amounts = [

        value

        for ts, value
        in sender_history[sender_id]

        if ts >= seven_days_ago
    ]

    if recent_amounts:

        avg_amount_7d = float(
            np.mean(recent_amounts)
        )

    else:

        # No previous history.
        # Use current amount as neutral baseline.
        avg_amount_7d = float(amount)

    amount_deviation = (
        amount
        - avg_amount_7d
    )

    return (
        sender_transactions_1h,
        sender_transactions_24h,
        receiver_transactions_1h,
        avg_amount_7d,
        amount_deviation
    )


# ============================================================
# TRANSACTION GENERATOR
# ============================================================

def generate_transaction(
    transaction_id,
    timestamp,
    update_state=True
):

    sender_id = choose_sender()

    sender_wallet = wallets[
        sender_id
    ]

    transaction_type = (
        get_transaction_type(
            sender_wallet["profile"]
        )
    )

    payment_mode = (
        get_payment_mode(
            transaction_type
        )
    )

    amount = generate_amount(
        transaction_type
    )

    # --------------------------------------------------------
    # RECEIVER + LOCATION
    # --------------------------------------------------------

    if transaction_type == "purchase":

        merchant_id = random.choice(
            merchant_ids
        )

        merchant = merchants[
            merchant_id
        ]

        receiver_id = merchant_id

        merchant_category = (
            merchant["category"]
        )

        receiver_balance = np.nan

        receiver_is_wallet = False

    else:

        possible_receivers = [
            w
            for w in wallet_ids
            if w != sender_id
        ]

        receiver_id = random.choice(
            possible_receivers
        )

        receiver_wallet = wallets[
            receiver_id
        ]

        merchant_category = "none"

        receiver_balance = (
            receiver_wallet["balance"]
        )

        receiver_is_wallet = True

    # --------------------------------------------------------
    # TRANSACTION LOCATION
    #
    # 93% home-country
    # 7% cross-border
    # --------------------------------------------------------

    if random.random() < 0.93:

        country = (
            sender_wallet[
                "home_country"
            ]
        )

    else:

        country = random.choice([

            c
            for c in COUNTRIES

            if c != sender_wallet[
                "home_country"
            ]
        ])

    # --------------------------------------------------------
    # INTERNATIONAL FLAG
    # --------------------------------------------------------

    is_international = int(

        country
        != sender_wallet[
            "home_country"
        ]
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    new_device = int(
        random.random() < 0.03
    )

    if new_device:

        device_id = generate_device_id()

    else:

        device_id = (
            sender_wallet["device_id"]
        )

    # --------------------------------------------------------
    # LOCATION CHANGE
    #
    # TRUE semantic definition:
    #
    # 1 = current transaction location differs
    #     from wallet's previous transaction location
    #
    # 0 = same location
    # --------------------------------------------------------

    new_location = int(

        country
        != sender_wallet[
            "last_location"
        ]
    )

    # --------------------------------------------------------
    # BEHAVIOURAL FEATURES
    # --------------------------------------------------------

    (
        transactions_1h,
        transactions_24h,
        receiver_transactions_1h,
        avg_amount_7d,
        amount_deviation
    ) = calculate_features(

        sender_id,
        receiver_id,
        timestamp,
        amount,
        receiver_is_wallet
    )

    # --------------------------------------------------------
    # BALANCE
    # --------------------------------------------------------

    sender_balance = (
        sender_wallet["balance"]
    )

    # --------------------------------------------------------
    # INSUFFICIENT FUNDS
    #
    # Adjust amount BEFORE final feature calculation
    # so amount_deviation stays consistent.
    # --------------------------------------------------------

    if amount > sender_balance:

        amount = round(

            max(
                1.0,
                sender_balance
                * random.uniform(
                    0.01,
                    0.25
                )
            ),

            2
        )

        # Recalculate deviation
        amount_deviation = (
            amount
            - avg_amount_7d
        )

    # --------------------------------------------------------
    # VELOCITY
    # --------------------------------------------------------

    sender_velocity = (
        transactions_1h
    )

    receiver_velocity = (
        receiver_transactions_1h
    )

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    row = {

        "transaction_id":
            transaction_id,

        "timestamp":
            timestamp,

        "sender_id":
            sender_id,

        "receiver_id":
            receiver_id,

        "amount_eur":
            amount,

        "transaction_type":
            transaction_type,

        "payment_mode":
            payment_mode,

        "psp_id":
            random.choice(PSPS),

        "country":
            country,

        "device_id":
            device_id,

        "merchant_category":
            merchant_category,

        "is_international":
            is_international,

        "sender_balance":
            round(
                sender_balance,
                2
            ),

        "receiver_balance":
            (
                round(
                    receiver_balance,
                    2
                )

                if not pd.isna(
                    receiver_balance
                )

                else np.nan
            ),

        "transactions_1h":
            int(transactions_1h),

        "transactions_24h":
            int(transactions_24h),

        "avg_amount_7d":
            round(
                avg_amount_7d,
                2
            ),

        "amount_deviation":
            round(
                amount_deviation,
                2
            ),

        "new_device":
            new_device,

        "new_location":
            new_location,

        "sender_velocity":
            int(sender_velocity),

        "receiver_velocity":
            int(receiver_velocity)
    }

    # --------------------------------------------------------
    # UPDATE STATE
    # --------------------------------------------------------

    if update_state:

        sender_wallet["balance"] = round(

            max(
                0,
                sender_wallet["balance"]
                - amount
            ),

            2
        )

        sender_wallet["device_id"] = (
            device_id
        )

        sender_wallet["last_location"] = (
            country
        )

        # Sender history retained for 7 days
        sender_history[
            sender_id
        ].append(
            (
                timestamp,
                amount
            )
        )

        # P2P receiver state
        if transaction_type == "p2p_transfer":

            wallets[
                receiver_id
            ]["balance"] = round(

                wallets[
                    receiver_id
                ]["balance"]
                + amount,

                2
            )

            receiver_history[
                receiver_id
            ].append(
                (
                    timestamp,
                    amount
                )
            )

    return row


# ============================================================
# WARM-UP
# ============================================================

print(
    f"Generating {WARMUP_DAYS}-day warm-up history..."
)

warmup_start = (
    pd.Timestamp(START_DATE)
    - pd.Timedelta(
        days=WARMUP_DAYS
    )
)

warmup_end = pd.Timestamp(
    START_DATE
)

for i in range(
    WARMUP_TRANSACTIONS
):

    timestamp = generate_timestamp(
        warmup_start,
        warmup_end
    )

    generate_transaction(

        transaction_id=
            f"WARM{i + 1:06d}",

        timestamp=timestamp,

        update_state=True
    )


# ============================================================
# ACTUAL DATASET
# ============================================================

print(
    f"Generating {N_TRANSACTIONS:,} transactions..."
)

rows = []

start = pd.Timestamp(
    START_DATE
)

end = pd.Timestamp(
    END_DATE
)

for i in range(
    N_TRANSACTIONS
):

    timestamp = generate_timestamp(
        start,
        end
    )

    row = generate_transaction(

        transaction_id=
            f"TX{i + 1:06d}",

        timestamp=timestamp,

        update_state=True
    )

    rows.append(row)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# DATA TYPES
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

integer_columns = [

    "is_international",
    "transactions_1h",
    "transactions_24h",
    "new_device",
    "new_location",
    "sender_velocity",
    "receiver_velocity"
]

for column in integer_columns:

    df[column] = (
        df[column]
        .fillna(0)
        .astype(int)
    )


float_columns = [

    "amount_eur",
    "sender_balance",
    "receiver_balance",
    "avg_amount_7d",
    "amount_deviation"
]

for column in float_columns:

    df[column] = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
        .round(2)
    )


# ============================================================
# ASSERTIONS
# ============================================================

assert len(df) == N_TRANSACTIONS

assert df["transaction_id"].is_unique

assert set(
    df["new_device"].unique()
).issubset({0, 1})

assert set(
    df["new_location"].unique()
).issubset({0, 1})

assert set(
    df["is_international"].unique()
).issubset({0, 1})

assert (
    df["timestamp"]
    .is_monotonic_increasing
)

assert (
    df["transactions_1h"] >= 0
).all()

assert (
    df["transactions_24h"] >= 0
).all()

assert (
    df["transactions_1h"]
    <= df["transactions_24h"]
).all()

assert (
    df["amount_eur"] > 0
).all()

assert (
    df["sender_balance"] >= 0
).all()

purchase_mask = (
    df["transaction_type"]
    == "purchase"
)

p2p_mask = (
    df["transaction_type"]
    == "p2p_transfer"
)

assert (
    df.loc[
        purchase_mask,
        "receiver_balance"
    ]
    .isna()
    .all()
)

assert (
    df.loc[
        p2p_mask,
        "receiver_balance"
    ]
    .notna()
    .all()
)

assert set(
    df["transaction_type"].unique()
).issubset(
    set(TRANSACTION_TYPES)
)

assert set(
    df["payment_mode"].unique()
).issubset(
    set(PAYMENT_MODES)
)


# ============================================================
# SCHEMA CHECK
# ============================================================

expected_columns = [

    "transaction_id",
    "timestamp",
    "sender_id",
    "receiver_id",
    "amount_eur",
    "transaction_type",
    "payment_mode",
    "psp_id",
    "country",
    "device_id",
    "merchant_category",
    "is_international",
    "sender_balance",
    "receiver_balance",
    "transactions_1h",
    "transactions_24h",
    "avg_amount_7d",
    "amount_deviation",
    "new_device",
    "new_location",
    "sender_velocity",
    "receiver_velocity"
]

assert list(
    df.columns
) == expected_columns


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_PATH
    ),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# VALIDATION REPORT
# ============================================================

print("\n")
print("=" * 50)
print("GENERATION COMPLETE")
print("=" * 50)

print(
    f"Transactions : {len(df):,}"
)

print(
    f"Wallets      : {N_WALLETS:,}"
)

print(
    f"Merchants    : {N_MERCHANTS:,}"
)

print(
    f"Saved to     : {OUTPUT_PATH}"
)


print("\nTransaction types:")
print(
    df["transaction_type"]
    .value_counts()
)


print("\nPayment modes:")
print(
    df["payment_mode"]
    .value_counts()
)


print("\nInternational ratio:")
print(
    df["is_international"]
    .value_counts(
        normalize=True
    )
)


print(
    "\nnew_device unique values:",
    sorted(
        df["new_device"]
        .unique()
        .tolist()
    )
)

print(
    "new_location unique values:",
    sorted(
        df["new_location"]
        .unique()
        .tolist()
    )
)


print("\nnew_device ratio:")
print(
    df["new_device"]
    .value_counts(
        normalize=True
    )
)


print("\nnew_location ratio:")
print(
    df["new_location"]
    .value_counts(
        normalize=True
    )
)


print("\nBehavioural features:")

behavior_columns = [

    "transactions_1h",
    "transactions_24h",
    "sender_velocity",
    "receiver_velocity"
]

print(
    df[
        behavior_columns
    ].describe()
)


print(
    "\nNon-zero behavioural ratios:"
)

for column in behavior_columns:

    ratio = (
        df[column]
        .gt(0)
        .mean()
        * 100
    )

    print(
        f"{column:<22}: "
        f"{ratio:.2f}% non-zero"
    )


print(
    "\nWallet transaction frequency:"
)

print(
    df["sender_id"]
    .value_counts()
    .describe()
)


print(
    "\nBalance sanity checks:"
)

print(
    "Negative sender_balance rows   :",
    (
        df["sender_balance"] < 0
    ).sum()
)

print(
    "Negative receiver_balance rows :",
    (
        df["receiver_balance"] < 0
    ).sum()
)

print(
    "Purchase rows with non-NaN receiver_balance :",
    df.loc[
        purchase_mask,
        "receiver_balance"
    ].notna().sum()
)

print(
    "P2P rows with NaN receiver_balance          :",
    df.loc[
        p2p_mask,
        "receiver_balance"
    ].isna().sum()
)


print(
    "\nDuplicate transaction IDs:",
    df["transaction_id"]
    .duplicated()
    .sum()
)

print(
    "Timestamp ordering OK:",
    df["timestamp"]
    .is_monotonic_increasing
)


print("\nMissing values:")
print(
    df.isna().sum()
)


print("\nData types:")
print(
    df.dtypes
)


print("\nPreview:")
print(
    df.head(10)
    .to_string(index=False)
)


print(
    "\nAll assertions passed. Done."
)