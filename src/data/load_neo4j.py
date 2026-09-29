import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE")

CSV_PATH = "data/processed/transactions_with_fraud.csv"

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df):,} transactions")

with driver.session(database=DATABASE) as session:
    session.run("""
        CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS
        FOR (t:Transaction)
        REQUIRE t.transaction_id IS UNIQUE
    """)

    session.run("""
        CREATE CONSTRAINT wallet_id_unique IF NOT EXISTS
        FOR (w:Wallet)
        REQUIRE w.wallet_id IS UNIQUE
    """)

    for start in range(0, len(df), 1000):
        batch = df.iloc[start:start + 1000].to_dict("records")

        session.run("""
            UNWIND $rows AS row

            MERGE (sender:Wallet {
                wallet_id: row.sender_id
            })

            MERGE (receiver:Wallet {
                wallet_id: row.receiver_id
            })

            CREATE (t:Transaction {
                transaction_id: row.transaction_id,
                amount_eur: row.amount_eur,
                timestamp: row.timestamp,
                transaction_type: row.transaction_type,
                payment_mode: row.payment_mode,
                is_fraud: row.is_fraud,
                transactions_1h: row.transactions_1h,
                transactions_24h: row.transactions_24h,
                sender_velocity: row.sender_velocity,
                receiver_velocity: row.receiver_velocity
            })

            CREATE (sender)-[:SENT]->(t)
            CREATE (t)-[:RECEIVED_BY]->(receiver)
        """, rows=batch)

        print(f"Loaded {min(start + 1000, len(df)):,} / {len(df):,}")

driver.close()

print("finished loading transactions into Neo4j")