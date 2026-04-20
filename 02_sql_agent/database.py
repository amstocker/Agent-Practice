"""
Database setup for the SQL agent.

Creates an in-memory SQLite database with a small music catalog:
- artists (electronic and other genres)
- albums
- tracks

The database is created once and shared via get_connection().
"""

import sqlite3

# Shared in-memory connection. In-memory databases only exist as long as
# the connection is open, so we keep a single global connection.
_connection = None


def get_connection() -> sqlite3.Connection:
    """Return the shared database connection, creating it on first call."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(":memory:")
        _connection.row_factory = sqlite3.Row  # allows column access by name
        _create_tables(_connection)
        _seed_data(_connection)
    return _connection


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE artists (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            genre TEXT NOT NULL
        );

        CREATE TABLE albums (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            artist_id INTEGER NOT NULL,
            release_year INTEGER NOT NULL,
            FOREIGN KEY (artist_id) REFERENCES artists(id)
        );

        CREATE TABLE tracks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            album_id INTEGER NOT NULL,
            duration_seconds INTEGER NOT NULL,
            FOREIGN KEY (album_id) REFERENCES albums(id)
        );
    """)


def _seed_data(conn: sqlite3.Connection):
    # Artists — mix of electronic and other genres
    conn.executemany("INSERT INTO artists (id, name, genre) VALUES (?, ?, ?)", [
        (1, "Aphex Twin", "Electronic"),
        (2, "Boards of Canada", "Electronic"),
        (3, "Radiohead", "Alternative Rock"),
        (4, "Burial", "Electronic"),
        (5, "Björk", "Art Pop"),
        (6, "Four Tet", "Electronic"),
    ])

    # Albums
    conn.executemany("INSERT INTO albums (id, title, artist_id, release_year) VALUES (?, ?, ?, ?)", [
        (1, "Selected Ambient Works 85-92", 1, 1992),
        (2, "Richard D. James Album", 1, 1996),
        (3, "Music Has the Right to Children", 2, 1998),
        (4, "Geogaddi", 2, 2002),
        (5, "OK Computer", 3, 1997),
        (6, "Kid A", 3, 2000),
        (7, "Untrue", 4, 2007),
        (8, "Homogenic", 5, 1997),
        (9, "Rounds", 6, 2003),
    ])

    # Tracks (a few per album)
    conn.executemany("INSERT INTO tracks (id, title, album_id, duration_seconds) VALUES (?, ?, ?, ?)", [
        # Selected Ambient Works 85-92
        (1, "Xtal", 1, 289),
        (2, "Tha", 1, 540),
        (3, "Pulsewidth", 1, 225),
        # Richard D. James Album
        (4, "4", 2, 228),
        (5, "Cornish Acid", 2, 98),
        # Music Has the Right to Children
        (6, "Roygbiv", 3, 145),
        (7, "Aquarius", 3, 360),
        # Geogaddi
        (8, "Music Is Math", 4, 340),
        (9, "1969", 4, 286),
        # OK Computer
        (10, "Paranoid Android", 5, 383),
        (11, "Karma Police", 5, 264),
        # Kid A
        (12, "Everything In Its Right Place", 6, 250),
        (13, "Idioteque", 6, 309),
        # Untrue
        (14, "Archangel", 7, 240),
        (15, "Near Dark", 7, 337),
        # Homogenic
        (16, "Jóga", 8, 305),
        (17, "Hunter", 8, 249),
        # Rounds
        (18, "Hands", 9, 286),
        (19, "She Moves She", 9, 350),
    ])

    conn.commit()
