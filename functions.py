import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Function for W2V version
def find_similar_embedding_W2V(word1, word2, embedded_words, embeddings, W2V_embedding):
    embedding1 = W2V_embedding[word1]
    embedding2 = W2V_embedding[word2]

    # Combining embeddings
    combined_embedding = embedding1 + embedding2

    # Computing cosine distance to embedded wordlist
    similarities = cosine_similarity([combined_embedding], embeddings)

    # gets the top 10 matches
    sorted_indices = np.argsort(similarities[0])[::-1][:10]  # Top 10 matches

    # Retrieve the top words and their corresponding similarity scores
    top_words = [embedded_words[i] for i in sorted_indices]
    top_similarities = [similarities[0][i] for i in sorted_indices]

    # Print top 10 closest words
    print(f"Top 10 closest words with cosine similarity for the sum of {word1} and {word2}:")
    for word, similarity in zip(top_words, top_similarities):
        print(f"{word}: {similarity:.4f}")

    # Selects the closest not-input word
    if top_words[0] not in [word1, word2]:
        top_word = top_words[0]
    elif top_words[1] not in [word1, word2]:
        top_word = top_words[1]
    else:
        top_word = top_words[2]


    return top_word

#Function for ST-version
def find_similar_embedding_ST(word1, word2, embedded_words, embeddings, ST_embedding):

    #Creating combined string
    combined_sentence = word1 + " and " + word2 + " combined"
    combined_embedding = ST_embedding.encode(combined_sentence)

    # Computing cosine distance to embedded wordlist
    similarities = cosine_similarity([combined_embedding], embeddings)

    # gets the top 10 matches
    sorted_indices = np.argsort(similarities[0])[::-1][:10]  # Top 10 matches

    # Retrieve the top words and their corresponding similarity scores
    top_words = [embedded_words[i] for i in sorted_indices]
    top_similarities = [similarities[0][i] for i in sorted_indices]

    # Print top 10 closest words
    print(f'Top 10 closest words for "{word1} combined with {word2}" cosine similarity:')
    for word, similarity in zip(top_words, top_similarities):
        print(f"{word}: {similarity:.4f}")

    # Selects the closest not-input word
    if top_words[0] not in [word1, word2]:
        top_word = top_words[0]
    elif top_words[1] not in [word1, word2]:
        top_word = top_words[1]
    else:
        top_word = top_words[2]

    return top_word