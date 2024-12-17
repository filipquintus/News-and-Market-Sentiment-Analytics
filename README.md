## **News and Market Sentiment Analysis - Exam**

This project contains two version of a crafting game as a part of the exam for the course *DS821: News and Market Sentiment Analytics*:
 - **W2V_game** which is build on a Word2Vec-model
 - **ST_game** which is build on a SentenceTransformer-model.


---

## **Prerequisites**


**Dependencies**:
   To runt the game, the following libraries must be installed:
   ```bash
   pip install pygame gensim scikit-learn sentence-transformers numpy kagglehub
   ```

**Word2Vec Model**:
    
   - The `GoogleNews-vectors-negative300.bin` file is required for the **W2V_game**. When running the scrip for the **W2V_game**, the model is automaticly downloaded using the following:
     ```python
     import kagglehub
     path = kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300")
     ```
   - After downloading, update the `path` variable in `W2V_game.py` to point to the local file.
   - If problem arises, the model can be downloaded manually from: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300
   - BE AWARE THAT THE GoogleNews-vectors-negative300 MODEL WHICH TAKES UP 3.64 GB


---

## **How to Run the Games**

1. **Download the Required Files**:
   - Ensure all `.npy` data files and the Word2Vec model are downloaded and accessible in the project directory.
   
2. **Run the SentenceTransformer Game**:
   ```bash
   python ST_game.py
   ```

3. **Run the Word2Vec Game**:
   ```bash
   python W2V_game.py
   ```


## **Acknowledgments**
- **Pygame**: Used for building the GUI.
- **Gensim**: For Word2Vec model operations.
- **Sentence-Transformers**: For SentenceTransformer embeddings.
- **Kaggle**: Source for pre-trained Word2Vec embeddings.


