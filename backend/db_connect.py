import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
    raise ValueError(
        "Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file."
    )

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)


def verify_connection():
    """Verify that the Neo4j server is reachable."""
    driver.verify_connectivity()


def close_connection():
    """Close the Neo4j driver when it is no longer needed."""
    driver.close()


if __name__ == "__main__":
    try:
        verify_connection()
        print("Connected to Neo4j successfully.")
    finally:
        close_connection()
