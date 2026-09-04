-- таблица музыкальных композиций
CREATE TABLE IF NOT EXISTS "tracks" (
    "track_id" TEXT PRIMARY KEY,
    "artist_name" TEXT NOT NULL,
    "track_name" TEXT NOT NULL,
    "genre" TEXT,
    "popularity" REAL DEFAULT 0.0 CHECK ("popularity" >= 0.0),
    "year" INTEGER,
    "valence" REAL NOT NULL CHECK ("valence" BETWEEN 0.0 AND 1.0),
    "energy" REAL NOT NULL CHECK ("energy" BETWEEN 0.0 AND 1.0),
    "danceability" REAL NOT NULL CHECK ("danceability" BETWEEN 0.0 AND 1.0),
    "acousticness" REAL NOT NULL CHECK ("acousticness" BETWEEN 0.0 AND 1.0),
    "instrumentalness" REAL NOT NULL CHECK ("instrumentalness" BETWEEN 0.0 AND 1.0),
    "key" TEXT,
    "loudness" REAL,
    "mode" TEXT,
    "speechiness" REAL,
    "liveness" REAL,
    "tempo" REAL,
    "duration_ms" INTEGER,
    "time_signature" TEXT
);

-- таблица сбора пользовательских оценок и параметров анализа
CREATE TABLE IF NOT EXISTS "ratings" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "timestamp" DATETIME DEFAULT CURRENT_TIMESTAMP,
    "rating" INTEGER CHECK ("rating" BETWEEN 1 AND 5),
    "valence" REAL NOT NULL CHECK ("valence" BETWEEN 0.0 AND 1.0),
    "energy" REAL NOT NULL CHECK ("energy" BETWEEN 0.0 AND 1.0),
    "danceability" REAL NOT NULL CHECK ("danceability" BETWEEN 0.0 AND 1.0),
    "acousticness" REAL NOT NULL CHECK ("acousticness" BETWEEN 0.0 AND 1.0),
    "instrumentalness" REAL NOT NULL CHECK ("instrumentalness" BETWEEN 0.0 AND 1.0),
    "hue" REAL NOT NULL,
    "saturation" REAL NOT NULL,
    "brightness" REAL NOT NULL,
    "description" TEXT,
    "tracks_json" TEXT NOT NULL
);

-- индексы для оптимизации селективности и ускорения выборки
CREATE INDEX IF NOT EXISTS "idx_tracks_genre" ON "tracks" ("genre");
CREATE INDEX IF NOT EXISTS "idx_tracks_popularity" ON "tracks" ("popularity" DESC);
CREATE INDEX IF NOT EXISTS "idx_ratings_timestamp" ON "ratings" ("timestamp");