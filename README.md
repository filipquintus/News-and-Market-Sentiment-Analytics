# Project: Semantic Word Combination Games

This project contains two interactive games, **ST_Game** and **W2V_Game**, that allow players to combine words from an inventory to generate new words based on semantic embeddings. Each game utilizes a different embedding model: **SentenceTransformer** and **Word2Vec**, respectively.

---

## Files Overview

### 1. `functions.py`
- Contains utility functions to compute similar embeddings:
  - `find_similar_embedding_W2V`: Uses a Word2Vec model.
  - `find_similar_embedding_ST`: Uses a SentenceTransformer model.

### 2. `ST_game.py`
- Implements a game using the SentenceTransformer model.
- Features include:
  - Combining inventory words to create new words.
  - GUI built with Pygame.
  - Embedding files used: `embedded_words_ST.npy`, `embeddings_ST.npy`.

### 3. `W2V_game.py`
- Implements a game using the Word2Vec model.
- Features include:
  - Combining inventory words to create new words.
  - GUI built with Pygame.
  - Embedding files used: `embedded_words_W2V.npy`, `embeddings_W2V.npy`.
  - Pre-trained Word2Vec model file: `GoogleNews-vectors-negative300.bin`.

### 4. Required Data Files
- **SentenceTransformer Game**:
  - `embedded_words_ST.npy`
  - `embeddings_ST.npy`
- **Word2Vec Game**:
  - `embedded_words_W2V.npy`
  - `embeddings_W2V.npy`
  - `GoogleNews-vectors-negative300.bin`

---

## Game Descriptions

### 1. ST_Game (SentenceTransformer)
- Combines inventory words using SentenceTransformer embeddings to create new words.
- **Features**:
  - Inventory system to manage items (e.g., "water," "fire").
  - Semantic combination of words using SentenceTransformer.
  - Interactive GUI for selecting and combining words.

### 2. W2V_Game (Word2Vec)
- Combines inventory words using Word2Vec embeddings to create new words.
- **Features**:
  - Inventory system to manage items (e.g., "earth," "wind").
  - Semantic combination of words using Word2Vec.
  - Interactive GUI for selecting and combining words.

---

## Prerequisites

1. **Python Version**: Ensure Python 3.8 or later is installed.
2. **Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install pygame gensim scikit-learn sentence-transformers numpy
   ```
3. **Data Files**: Ensure all .npy embedding files are in the same directory as the scripts.
Update the path variable in W2V_game.py to point to the GoogleNews-vectors-negative300.bin file.
