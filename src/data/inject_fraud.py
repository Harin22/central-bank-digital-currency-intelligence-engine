import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SEED = 42

INPUT_PATH = Path("data/raw/normal_transactions.csv")
OUTPUT_PATH = Path("data/processed/transactions_with_fraud.csv")

rng = np.random.default_rng(SEED)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = (
    df.sort_values("timestamp")
      .reset_index(drop=True)
)

df["is_fraud"] = 0

print(f"Loaded transactions: {len(df):,}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def mark_fraud(indices):
    """
    Mark selected transactions as fraudulent.
    """
    indices = list(set(indices))

    df.loc[
        indices,
        "is_fraud"
    ] = 1

    return len(indices)


def get_clean_indices():
    """
    Return transactions that have not already
    been used by another fraud pattern.
    """
    return df.index[
        df["is_fraud"] == 0
    ].to_numpy()


def get_wallet_rows(wallet):
    """
    Return transaction indices for a sender wallet,
    ordered chronologically.
    """
    return (
        df[
            df["sender_id"] == wallet
        ]
        .sort_values("timestamp")
        .index
        .to_numpy()
    )


# ============================================================
# 1. ACCOUNT TAKEOVER
# ============================================================

def inject_account_takeover(target=2_000):

    print("\n[1] Account takeover")

    wallet_counts = df["sender_id"].value_counts()

    eligible_wallets = wallet_counts[
        wallet_counts >= 8
    ].index.to_numpy()

    rng.shuffle(eligible_wallets)

    selected_indices = []

    for wallet in eligible_wallets:

        wallet_rows = get_wallet_rows(wallet)

        # Only use currently clean transactions
        wallet_rows = wallet_rows[
            df.loc[
                wallet_rows,
                "is_fraud"
            ].to_numpy() == 0
        ]

        if len(wallet_rows) < 8:
            continue

        # ----------------------------------------------------
        # Select a short sequence
        # ----------------------------------------------------

        max_sequence_size = min(
            7,
            len(wallet_rows)
        )

        sequence_size = int(
            rng.integers(
                3,
                max_sequence_size + 1
            )
        )

        max_start = (
            len(wallet_rows)
            - sequence_size
        )

        if max_start < 0:
            continue

        start = int(
            rng.integers(
                0,
                max_start + 1
            )
        )

        sequence = wallet_rows[
            start:start + sequence_size
        ]

        selected_indices.extend(
            sequence.tolist()
        )

        if len(selected_indices) >= target:
            break

    selected_indices = selected_indices[:target]

    # ========================================================
    # CREATE ACCOUNT TAKEOVER SIGNALS
    # ========================================================

    for idx in selected_indices:

        # New device
        df.loc[
            idx,
            "new_device"
        ] = 1

        # New location
        df.loc[
            idx,
            "new_location"
        ] = 1

        # ----------------------------------------------------
        # Unusual transaction amount
        # ----------------------------------------------------

        original_amount = df.loc[
            idx,
            "amount_eur"
        ]

        multiplier = rng.uniform(
            5.0,
            15.0
        )

        new_amount = np.clip(
            original_amount * multiplier,
            100,
            5000
        )

        df.loc[
            idx,
            "amount_eur"
        ] = round(
            new_amount,
            2
        )

        # ----------------------------------------------------
        # Amount deviation
        # ----------------------------------------------------

        avg_amount = df.loc[
            idx,
            "avg_amount_7d"
        ]

        if (
            pd.notna(avg_amount)
            and avg_amount > 0
        ):

            df.loc[
                idx,
                "amount_deviation"
            ] = round(
                new_amount - avg_amount,
                2
            )

        # ----------------------------------------------------
        # Some takeover transactions also become fast
        # ----------------------------------------------------

        df.loc[
            idx,
            "transactions_1h"
        ] = max(
            int(df.loc[idx, "transactions_1h"]),
            int(rng.integers(5, 15))
        )

        df.loc[
            idx,
            "sender_velocity"
        ] = max(
            int(df.loc[idx, "sender_velocity"]),
            int(rng.integers(5, 15))
        )

    count = mark_fraud(
        selected_indices
    )

    print(
        f"   Fraud transactions: {count:,}"
    )

    return count


# ============================================================
# 2. HIGH-FREQUENCY BURST
# ============================================================

def inject_burst(target=2_000):

    print("\n[2] High-frequency burst")

    wallet_counts = df["sender_id"].value_counts()

    eligible_wallets = wallet_counts[
        wallet_counts >= 10
    ].index.to_numpy()

    rng.shuffle(eligible_wallets)

    selected_indices = []

    for wallet in eligible_wallets:

        wallet_rows = get_wallet_rows(wallet)

        # Only clean transactions
        wallet_rows = wallet_rows[
            df.loc[
                wallet_rows,
                "is_fraud"
            ].to_numpy() == 0
        ]

        if len(wallet_rows) < 10:
            continue

        # ----------------------------------------------------
        # Select burst sequence
        # ----------------------------------------------------

        max_sequence_size = min(
            11,
            len(wallet_rows)
        )

        sequence_size = int(
            rng.integers(
                5,
                max_sequence_size + 1
            )
        )

        max_start = (
            len(wallet_rows)
            - sequence_size
        )

        if max_start < 0:
            continue

        start = int(
            rng.integers(
                0,
                max_start + 1
            )
        )

        sequence = wallet_rows[
            start:start + sequence_size
        ]

        selected_indices.extend(
            sequence.tolist()
        )

        if len(selected_indices) >= target:
            break

    selected_indices = selected_indices[:target]

    # ========================================================
    # CREATE BURST SIGNALS
    # ========================================================

    for idx in selected_indices:

        # Extremely high short-term activity
        df.loc[
            idx,
            "transactions_1h"
        ] = int(
            rng.integers(
                15,
                40
            )
        )

        # High daily activity
        df.loc[
            idx,
            "transactions_24h"
        ] = int(
            rng.integers(
                30,
                80
            )
        )

        # High sender velocity
        df.loc[
            idx,
            "sender_velocity"
        ] = int(
            rng.integers(
                15,
                40
            )
        )

    count = mark_fraud(
        selected_indices
    )

    print(
        f"   Fraud transactions: {count:,}"
    )

    return count


# ============================================================
# 3. SMURFING / STRUCTURING
# ============================================================

def inject_smurfing(target=2_000):

    print("\n[3] Smurfing / structuring")

    wallet_counts = df["sender_id"].value_counts()

    eligible_wallets = wallet_counts[
        wallet_counts >= 8
    ].index.to_numpy()

    rng.shuffle(eligible_wallets)

    selected_indices = []

    for wallet in eligible_wallets:

        wallet_rows = get_wallet_rows(wallet)

        # Only clean transactions
        wallet_rows = wallet_rows[
            df.loc[
                wallet_rows,
                "is_fraud"
            ].to_numpy() == 0
        ]

        if len(wallet_rows) < 8:
            continue

        # ----------------------------------------------------
        # Select several transactions from same wallet
        # ----------------------------------------------------

        max_sequence_size = min(
            8,
            len(wallet_rows)
        )

        sequence_size = int(
            rng.integers(
                4,
                max_sequence_size + 1
            )
        )

        max_start = (
            len(wallet_rows)
            - sequence_size
        )

        if max_start < 0:
            continue

        start = int(
            rng.integers(
                0,
                max_start + 1
            )
        )

        sequence = wallet_rows[
            start:start + sequence_size
        ]

        selected_indices.extend(
            sequence.tolist()
        )

        if len(selected_indices) >= target:
            break

    selected_indices = selected_indices[:target]

    # ========================================================
    # CREATE SMURFING SIGNALS
    # ========================================================

    for idx in selected_indices:

        # ----------------------------------------------------
        # Repeated small transfers
        # ----------------------------------------------------

        new_amount = rng.uniform(
            20,
            150
        )

        df.loc[
            idx,
            "amount_eur"
        ] = round(
            new_amount,
            2
        )

        # ----------------------------------------------------
        # Increased transaction frequency
        # ----------------------------------------------------

        df.loc[
            idx,
            "transactions_1h"
        ] = int(
            rng.integers(
                5,
                15
            )
        )

        df.loc[
            idx,
            "transactions_24h"
        ] = int(
            rng.integers(
                15,
                35
            )
        )

        df.loc[
            idx,
            "sender_velocity"
        ] = int(
            rng.integers(
                5,
                15
            )
        )

        # ----------------------------------------------------
        # Amount deviation
        # ----------------------------------------------------

        avg_amount = df.loc[
            idx,
            "avg_amount_7d"
        ]

        if (
            pd.notna(avg_amount)
            and avg_amount > 0
        ):

            df.loc[
                idx,
                "amount_deviation"
            ] = round(
                new_amount - avg_amount,
                2
            )

    count = mark_fraud(
        selected_indices
    )

    print(
        f"   Fraud transactions: {count:,}"
    )

    return count


# ============================================================
# 4. MULE CHAIN
# ============================================================

def inject_mule_chain(target=2_000):

    print("\n[4] Mule-chain behaviour")

    # --------------------------------------------------------
    # Select wallets with enough activity
    # --------------------------------------------------------

    wallet_counts = df["sender_id"].value_counts()

    eligible_wallets = wallet_counts[
        wallet_counts >= 6
    ].index.to_numpy()

    rng.shuffle(eligible_wallets)

    selected_indices = []

    # --------------------------------------------------------
    # Build wallet groups
    #
    # A → B → C → D
    # --------------------------------------------------------

    for i in range(
        0,
        len(eligible_wallets) - 3,
        4
    ):

        if len(selected_indices) >= target:
            break

        wallet_a = eligible_wallets[i]
        wallet_b = eligible_wallets[i + 1]
        wallet_c = eligible_wallets[i + 2]
        wallet_d = eligible_wallets[i + 3]

        wallets = [
            wallet_a,
            wallet_b,
            wallet_c,
            wallet_d
        ]

        chain_rows = []

        valid_chain = True

        # ----------------------------------------------------
        # Select one or more CLEAN transactions per wallet
        # ----------------------------------------------------

        for wallet in wallets:

            wallet_rows = get_wallet_rows(wallet)

            wallet_rows = wallet_rows[
                df.loc[
                    wallet_rows,
                    "is_fraud"
                ].to_numpy() == 0
            ]

            if len(wallet_rows) < 2:
                valid_chain = False
                break

            # Pick recent clean transactions
            rows = wallet_rows[-2:]

            chain_rows.append(rows)

        if not valid_chain:
            continue

        # ----------------------------------------------------
        # Create actual receiver chain
        #
        # A → B
        # B → C
        # C → D
        # D → C
        # ----------------------------------------------------

        receivers = [
            wallet_b,
            wallet_c,
            wallet_d,
            wallet_c
        ]

        for rows, receiver in zip(
            chain_rows,
            receivers
        ):

            for idx in rows:

                # Actual wallet-to-wallet movement
                df.loc[
                    idx,
                    "receiver_id"
                ] = receiver

                # Suspicious transfer amount
                new_amount = rng.uniform(
                    300,
                    1500
                )

                df.loc[
                    idx,
                    "amount_eur"
                ] = round(
                    new_amount,
                    2
                )

                # Increased activity
                df.loc[
                    idx,
                    "transactions_1h"
                ] = int(
                    rng.integers(
                        5,
                        15
                    )
                )

                df.loc[
                    idx,
                    "transactions_24h"
                ] = int(
                    rng.integers(
                        15,
                        35
                    )
                )

                df.loc[
                    idx,
                    "sender_velocity"
                ] = int(
                    rng.integers(
                        5,
                        15
                    )
                )

                # ------------------------------------------------
                # Amount deviation
                # ------------------------------------------------

                avg_amount = df.loc[
                    idx,
                    "avg_amount_7d"
                ]

                if (
                    pd.notna(avg_amount)
                    and avg_amount > 0
                ):

                    df.loc[
                        idx,
                        "amount_deviation"
                    ] = round(
                        new_amount - avg_amount,
                        2
                    )

                selected_indices.append(idx)

        if len(selected_indices) >= target:
            break

    selected_indices = list(
        dict.fromkeys(
            selected_indices
        )
    )

    selected_indices = selected_indices[:target]

    count = mark_fraud(
        selected_indices
    )

    print(
        f"   Fraud transactions: {count:,}"
    )

    return count


# ============================================================
# 5. UNUSUAL SPENDING BEHAVIOUR
# ============================================================

def inject_unusual_spending(target=1500):

    print("\n[5] Unusual spending behaviour")

    rng_local = np.random.default_rng(SEED + 5)

    clean_indices = get_clean_indices()

    if len(clean_indices) < target:
        raise ValueError("Not enough clean transactions for unusual spending.")

    # Prefer wallets with enough transaction history
    wallet_counts = df.loc[clean_indices, "sender_id"].value_counts()
    eligible_wallets = wallet_counts[wallet_counts >= 5].index.tolist()

    rng_local.shuffle(eligible_wallets)

    selected = []

    for wallet in eligible_wallets:

        if len(selected) >= target:
            break

        wallet_rows = df[
            (df["sender_id"] == wallet) &
            (df["is_fraud"] == 0)
        ].index.tolist()

        if len(wallet_rows) < 5:
            continue

        # Pick one transaction from this wallet
        idx = rng_local.choice(wallet_rows)

        selected.append(idx)

    # Safety fallback
    if len(selected) < target:

        remaining = list(
            set(clean_indices) - set(selected)
        )

        extra = rng_local.choice(
            remaining,
            size=target - len(selected),
            replace=False
        )

        selected.extend(extra.tolist())

    selected = selected[:target]

    for idx in selected:

        wallet = df.at[idx, "sender_id"]

        wallet_history = df[
            (df["sender_id"] == wallet) &
            (df["is_fraud"] == 0)
        ]

        # Typical wallet spending behaviour
        if len(wallet_history) > 0:

            normal_mean = wallet_history["amount_eur"].mean()
            normal_std = wallet_history["amount_eur"].std()

            if pd.isna(normal_std) or normal_std == 0:
                normal_std = max(normal_mean * 0.25, 10)

        else:
            normal_mean = 100
            normal_std = 25

        # Create a transaction far outside normal behaviour
        multiplier = rng_local.uniform(6, 15)

        new_amount = normal_mean * multiplier

        new_amount = np.clip(
            new_amount,
            500,
            5000
        )

        df.at[idx, "amount_eur"] = round(float(new_amount), 2)

        # Strong spending deviation
        df.at[idx, "amount_deviation"] = round(
            float((new_amount - normal_mean) / normal_std),
            2
        )

        # Sometimes combine with international/new location behaviour
        if rng_local.random() < 0.45:
            df.at[idx, "is_international"] = 1

        if rng_local.random() < 0.35:
            df.at[idx, "new_location"] = 1

        if rng_local.random() < 0.25:
            df.at[idx, "new_device"] = 1

    count = mark_fraud(selected)

    print(f"   Fraud transactions: {count:,}")

    return count


# ============================================================
# 6. CIRCULAR TRANSFERS
# ============================================================

def inject_circular_transfers(target=500):

    print("\n[6] Circular transfers")

    rng_local = np.random.default_rng(SEED + 6)

    clean_indices = get_clean_indices()

    # Get wallets with enough transactions
    wallet_counts = df.loc[clean_indices, "sender_id"].value_counts()

    eligible_wallets = wallet_counts[
        wallet_counts >= 2
    ].index.tolist()

    rng_local.shuffle(eligible_wallets)

    used_wallets = set()
    selected = []

    for i in range(0, len(eligible_wallets) - 3, 4):

        if len(selected) >= target:
            break

        wallets = eligible_wallets[i:i + 4]

        if len(wallets) < 4:
            break

        # Avoid reusing wallets
        if any(wallet in used_wallets for wallet in wallets):
            continue

        a, b, c, d = wallets

        # Find clean transactions for each wallet
        rows_a = df[
            (df["sender_id"] == a) &
            (df["is_fraud"] == 0)
        ].index.tolist()

        rows_b = df[
            (df["sender_id"] == b) &
            (df["is_fraud"] == 0)
        ].index.tolist()

        rows_c = df[
            (df["sender_id"] == c) &
            (df["is_fraud"] == 0)
        ].index.tolist()

        rows_d = df[
            (df["sender_id"] == d) &
            (df["is_fraud"] == 0)
        ].index.tolist()

        if not rows_a or not rows_b or not rows_c or not rows_d:
            continue

        idx_a = rng_local.choice(rows_a)
        idx_b = rng_local.choice(rows_b)
        idx_c = rng_local.choice(rows_c)
        idx_d = rng_local.choice(rows_d)

        indices = [
            idx_a,
            idx_b,
            idx_c,
            idx_d
        ]

        # Make sure all are still clean
        if any(df.at[idx, "is_fraud"] == 1 for idx in indices):
            continue

        # ----------------------------------------------------
        # Circular money movement:
        #
        # A → B
        # B → C
        # C → D
        # D → A
        # ----------------------------------------------------

        amount = round(
            float(rng_local.uniform(300, 1500)),
            2
        )

        df.at[idx_a, "receiver_id"] = b
        df.at[idx_b, "receiver_id"] = c
        df.at[idx_c, "receiver_id"] = d
        df.at[idx_d, "receiver_id"] = a

        # Similar amounts make the circular movement stronger
        amounts = [
            amount,
            round(amount * rng_local.uniform(0.90, 1.05), 2),
            round(amount * rng_local.uniform(0.90, 1.05), 2),
            round(amount * rng_local.uniform(0.90, 1.05), 2)
        ]

        for idx, tx_amount in zip(indices, amounts):

            df.at[idx, "amount_eur"] = tx_amount

            # Increase velocity signals
            df.at[idx, "transactions_1h"] = int(
                rng_local.integers(5, 15)
            )

            df.at[idx, "transactions_24h"] = int(
                rng_local.integers(15, 35)
            )

            df.at[idx, "sender_velocity"] = int(
                rng_local.integers(5, 15)
            )

            # Make behaviour unusual
            df.at[idx, "amount_deviation"] = round(
                float(rng_local.uniform(2.5, 7.0)),
                2
            )

        selected.extend(indices)

        used_wallets.update(wallets)

    # --------------------------------------------------------
    # Fallback if not enough full cycles were formed
    # --------------------------------------------------------

    selected = selected[:target]

    if len(selected) < target:

        remaining = list(
            set(clean_indices) - set(selected)
        )

        extra_needed = target - len(selected)

        if len(remaining) < extra_needed:
            raise ValueError(
                "Not enough clean transactions for circular transfers."
            )

        extra = rng_local.choice(
            remaining,
            size=extra_needed,
            replace=False
        )

        for idx in extra:

            df.at[idx, "transactions_1h"] = int(
                rng_local.integers(5, 15)
            )

            df.at[idx, "transactions_24h"] = int(
                rng_local.integers(15, 35)
            )

            df.at[idx, "sender_velocity"] = int(
                rng_local.integers(5, 15)
            )

            df.at[idx, "amount_deviation"] = round(
                float(rng_local.uniform(2.5, 7.0)),
                2
            )

        selected.extend(extra.tolist())

    count = mark_fraud(selected)

    print(f"   Fraud transactions: {count:,}")

    return count


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset():

    print("\n" + "=" * 60)
    print("FRAUD DATASET VALIDATION")
    print("=" * 60)

    total = len(df)

    fraud = int(
        df["is_fraud"].sum()
    )

    normal = total - fraud

    print(
        f"Total transactions : {total:,}"
    )

    print(
        f"Fraud transactions : {fraud:,}"
    )

    print(
        f"Normal transactions: {normal:,}"
    )

    print(
        f"Fraud percentage   : "
        f"{fraud / total * 100:.2f}%"
    )

    # ========================================================
    # FRAUD SIGNAL CHECKS
    # ========================================================

    print("\nFraud signal checks:")

    fraud_df = df[
        df["is_fraud"] == 1
    ]

    print(
        f"New device = 1    : "
        f"{(fraud_df['new_device'] == 1).mean() * 100:.2f}%"
    )

    print(
        f"New location = 1  : "
        f"{(fraud_df['new_location'] == 1).mean() * 100:.2f}%"
    )

    print(
        f"High 1h velocity  : "
        f"{(fraud_df['transactions_1h'] >= 15).mean() * 100:.2f}%"
    )

    print(
        f"High 24h velocity : "
        f"{(fraud_df['transactions_24h'] >= 30).mean() * 100:.2f}%"
    )

    print(
        f"Small transactions: "
        f"{(fraud_df['amount_eur'] <= 150).mean() * 100:.2f}%"
    )

    print(
        f"Large transactions: "
        f"{(fraud_df['amount_eur'] >= 300).mean() * 100:.2f}%"
    )

    # ========================================================
    # DATA INTEGRITY
    # ========================================================

    print("\nData integrity:")

    print(
        "Duplicate transaction IDs:",
        df["transaction_id"].duplicated().sum()
    )

    print(
        "Invalid amounts:",
        (df["amount_eur"] <= 0).sum()
    )

    print(
        "Invalid fraud labels:",
        (~df["is_fraud"].isin([0, 1])).sum()
    )

    print(
        "Missing transaction IDs:",
        df["transaction_id"].isna().sum()
    )

    print(
        "Missing sender IDs:",
        df["sender_id"].isna().sum()
    )

    print(
        "Missing receiver IDs:",
        df["receiver_id"].isna().sum()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inject_account_takeover(
        target=2_000
    )

    inject_burst(
        target=2_000
    )

    inject_smurfing(
        target=2_000
    )

    inject_mule_chain(
        target=2_000
    )

    inject_unusual_spending(
        target=1_500
    )

    inject_circular_transfers(
        target=500
    )

    validate_dataset()

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)