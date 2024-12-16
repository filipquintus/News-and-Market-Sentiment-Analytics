import pygame
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from functions import find_similar_embedding_W2V
from sentence_transformers import SentenceTransformer
import pandas as pd


# Load large Word2Vec model:
# Define the path to the Word2Vec binary file

## path = kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300")
### untag when submitting ##

path = "/Users/filipwillesen/.cache/kagglehub/datasets/leadbest/googlenewsvectorsnegative300/versions/2/GoogleNews-vectors-negative300.bin"
# Load the pre-trained Word2Vec model
W2V_embedding = KeyedVectors.load_word2vec_format(path, binary=True)


# loading words and embeddings

embedded_words = np.load("word_list/embedded_words_W2V.npy")
embeddings = np.load("word_list/embeddings_W2V.npy")


# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
GREY = (230, 230, 230)
RED = (232, 93, 93)
BLACK = (0, 0, 0)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Craft game - Word2Vec")

# Fonts
font = pygame.font.Font(None, 36)
font_message = pygame.font.Font(None, 24)
font_inventory = pygame.font.Font(None, 20)
font_reset = pygame.font.Font(None, 18)

# Initial inventory
inventory = ["water", "wind", "earth", "fire"]
selected_boxes = [None, None]
message = ""

# Box dimensions
BOX_WIDTH = 100
BOX_HEIGHT = 30

# Scroll position for inventory
SCROLL_OFFSET = 0
SCROLL_STEP = 10

# Function to draw a box
def draw_box(surface, text, x, y, color=BLACK):
    pygame.draw.rect(surface, color, (x, y, BOX_WIDTH, BOX_HEIGHT), 2)
    text_surface = font_inventory.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + BOX_WIDTH // 2, y + BOX_HEIGHT // 2))
    surface.blit(text_surface, text_rect)



# Main loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos

            # Check for clicks on inventory boxes
            for i, item in enumerate(inventory):
                box_x = SCREEN_WIDTH - 150
                box_y = 50 + i * (BOX_HEIGHT + 10) - SCROLL_OFFSET
                if box_x <= mouse_x <= box_x + BOX_WIDTH and box_y <= mouse_y <= box_y + BOX_HEIGHT:
                    # Add the selected box to the left or right side
                    if selected_boxes[0] is None:
                        selected_boxes[0] = item
                    elif selected_boxes[1] is None:
                        selected_boxes[1] = item

            # Combine strings if both boxes are selected
            if selected_boxes[0] and selected_boxes[1]:
                new_box = find_similar_embedding_W2V(selected_boxes[0], selected_boxes[1], embedded_words, embeddings, W2V_embedding)

                if new_box not in inventory:
                    inventory.append(new_box)
                message = f"By combining {selected_boxes[0]} and {selected_boxes[1]} you created {new_box}"
                selected_boxes = [None, None]


        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                inventory = ["water", "wind", "earth", "fire"]  # Reset inventory to initial state
                selected_boxes = [None, None]  # Clear selected boxes
                message = ""  # Clear message

        if event.type == pygame.MOUSEWHEEL:
            SCROLL_OFFSET += event.y * SCROLL_STEP
            SCROLL_OFFSET = max(0, SCROLL_OFFSET)  # Prevent scrolling above the top

    # Clear the screen
    screen.fill(GREY)

    # Draw main text
    main_text = font.render("Combine elements from your inventory", True, BLACK)
    screen.blit(main_text, (90, 50))
    reset_text = font_reset.render("press [r] to reset inventory", True, BLACK)
    screen.blit(reset_text, (450, 580))

    # Draw selected boxes
    if selected_boxes[0]:
        draw_box(screen, selected_boxes[0], 300, 200)
    if selected_boxes[1]:
        draw_box(screen, selected_boxes[1], 400, 150)

    # Draw inventory background
    pygame.draw.rect(screen, RED, (SCREEN_WIDTH - 150, 0, 150, SCREEN_HEIGHT))

    # Draw inventory title
    inventory_text = font.render("Inventory", True, BLACK)
    screen.blit(inventory_text, (SCREEN_WIDTH - 150 + 10, 10))

    # Draw inventory boxes with scroll offset
    for i, item in enumerate(inventory):
        box_x = SCREEN_WIDTH - 150 + 10
        box_y = 50 + i * (BOX_HEIGHT + 10) - SCROLL_OFFSET
        if 0 <= box_y < SCREEN_HEIGHT:  # Only draw visible boxes
            draw_box(screen, item, box_x, box_y)

    # Draw message
    if message:
        message_surface = font_message.render(message, True, BLACK)
        screen.blit(message_surface, (100, 350))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()