import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import numpy as np
import math
import time
import random
import json
import os


# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_INDEX = 0
IMG_SIZE = 300
OFFSET = 20

MODEL_PATH = r"C:\Users\Titif\OneDrive\Documents\Sign game\model\keras_model.h5"
LABELS_PATH = r"C:\Users\Titif\OneDrive\Documents\Sign game\model\labels.txt"

LABELS = ["1", "2", "3", "4", "5", "10"]

RECORD_FILE = "players.json"

TOTAL_ROUNDS = 5
ROUND_TIME = 5

POINTS_PER_SECOND = 10
COMBO_BONUS = 5


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================================================
# HAND DETECTOR
# =========================================================

detector = HandDetector(maxHands=1)


# =========================================================
# CLASSIFIER
# =========================================================

classifier = Classifier(
    MODEL_PATH,
    LABELS_PATH
)


# =========================================================
# LOAD SAVED RECORDS
# =========================================================

def load_records():

    if not os.path.exists(RECORD_FILE):
        return []

    try:

        with open(RECORD_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except:

        return []


# =========================================================
# SAVE RECORD
# =========================================================

def save_player(name, score):

    records = load_records()

    player = {
        "name": name,
        "score": score,
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    records.append(player)

    # Highest scores first
    records.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    with open(RECORD_FILE, "w", encoding="utf-8") as file:

        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# GET TOP PLAYERS
# =========================================================

def get_top_players():

    records = load_records()

    return records[:5]


# =========================================================
# DRAW CENTERED TEXT
# =========================================================

def draw_center_text(
    img,
    text,
    y,
    scale=2,
    color=(255, 255, 255),
    thickness=3
):

    font = cv2.FONT_HERSHEY_COMPLEX

    text_size = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )[0]

    x = (img.shape[1] - text_size[0]) // 2

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness
    )


# =========================================================
# DRAW HEADER
# =========================================================

def draw_header(img, title):

    cv2.rectangle(
        img,
        (0, 0),
        (img.shape[1], 90),
        (25, 25, 35),
        -1
    )

    cv2.putText(
        img,
        title,
        (30, 60),
        cv2.FONT_HERSHEY_COMPLEX,
        1.5,
        (255, 255, 255),
        3
    )


# =========================================================
# WELCOME SCREEN
# =========================================================

def welcome_screen():

    name = ""

    while True:

        success, img = cap.read()

        if not success:
            return None

        img = cv2.flip(img, 1)

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (150, 100),
            (1130, 620),
            (20, 20, 30),
            -1
        )

        img = cv2.addWeighted(
            overlay,
            0.85,
            img,
            0.15,
            0
        )

        draw_center_text(
            img,
            "SIGN CHALLENGE",
            190,
            2,
            (255, 255, 255),
            4
        )

        draw_center_text(
            img,
            "Enter player name",
            280,
            1,
            (180, 180, 180),
            2
        )

        cv2.rectangle(
            img,
            (350, 320),
            (930, 390),
            (50, 50, 65),
            -1
        )

        cv2.putText(
            img,
            name + "_",
            (370, 370),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2
        )

        draw_center_text(
            img,
            "Press ENTER to start",
            470,
            1,
            (100, 220, 100),
            2
        )

        draw_center_text(
            img,
            "ESC = Exit",
            530,
            0.8,
            (150, 150, 150),
            2
        )

        cv2.imshow(
            "Sign Challenge",
            img
        )

        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            return None

        elif key == 8:

            # Backspace
            name = name[:-1]

        elif key == 13:

            if name.strip():

                return name.strip()

        elif 32 <= key <= 126:

            if len(name) < 20:
                name += chr(key)


# =========================================================
# COUNTDOWN
# =========================================================

def countdown():

    start = time.time()

    while True:

        success, img = cap.read()

        if not success:
            return False

        img = cv2.flip(img, 1)

        elapsed = time.time() - start

        number = 3 - int(elapsed)

        if number <= 0:
            return True

        draw_header(
            img,
            "GET READY"
        )

        draw_center_text(
            img,
            str(number),
            430,
            6,
            (100, 220, 255),
            8
        )

        cv2.imshow(
            "Sign Challenge",
            img
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False


# =========================================================
# GAME ROUND
# =========================================================

def play_round(
    player_name,
    round_number,
    total_score
):

    target = random.choice(LABELS)

    round_score = 0

    correct_time = 0

    combo = 0

    max_combo = 0

    start_time = time.time()

    last_time = start_time

    prediction = "?"

    while True:

        success, img = cap.read()

        if not success:
            return total_score, False

        img = cv2.flip(img, 1)

        imgOutput = img.copy()

        now = time.time()

        elapsed = now - start_time

        delta = now - last_time

        last_time = now

        remaining = max(
            0,
            ROUND_TIME - elapsed
        )

        # -------------------------------------------------
        # ROUND FINISHED
        # -------------------------------------------------

        if elapsed >= ROUND_TIME:

            return (
                total_score + round_score,
                True
            )

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------

        draw_header(
            imgOutput,
            f"PLAYER: {player_name}"
        )

        # -------------------------------------------------
        # ROUND
        # -------------------------------------------------

        cv2.putText(
            imgOutput,
            f"ROUND {round_number}/{TOTAL_ROUNDS}",
            (930, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2
        )

        # -------------------------------------------------
        # TARGET NUMBER
        # -------------------------------------------------

        cv2.rectangle(
            imgOutput,
            (30, 115),
            (350, 310),
            (35, 35, 50),
            -1
        )

        cv2.putText(
            imgOutput,
            "TARGET",
            (80, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (180, 180, 180),
            2
        )

        cv2.putText(
            imgOutput,
            target,
            (145, 270),
            cv2.FONT_HERSHEY_COMPLEX,
            3,
            (100, 220, 255),
            5
        )

        # -------------------------------------------------
        # TIMER
        # -------------------------------------------------

        cv2.putText(
            imgOutput,
            f"TIME: {remaining:.1f}",
            (30, 370),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        cv2.putText(
            imgOutput,
            f"ROUND: +{round_score}",
            (30, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (100, 220, 100),
            3
        )

        cv2.putText(
            imgOutput,
            f"TOTAL: {total_score + round_score}",
            (30, 480),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

        # -------------------------------------------------
        # CAMERA / HAND DETECTION
        # -------------------------------------------------

        hands, img = detector.findHands(
            img,
            draw=False
        )

        if hands:

            hand = hands[0]

            x, y, w, h = hand["bbox"]

            y1 = max(
                0,
                y - OFFSET
            )

            y2 = min(
                img.shape[0],
                y + h + OFFSET
            )

            x1 = max(
                0,
                x - OFFSET
            )

            x2 = min(
                img.shape[1],
                x + w + OFFSET
            )

            imgCrop = img[
                y1:y2,
                x1:x2
            ]

            if imgCrop.size != 0 and w > 0 and h > 0:

                imgWhite = np.ones(
                    (IMG_SIZE, IMG_SIZE, 3),
                    np.uint8
                ) * 255

                aspectRatio = h / w

                # -------------------------------------------------
                # VERTICAL
                # -------------------------------------------------

                if aspectRatio > 1:

                    k = IMG_SIZE / h

                    wCal = math.ceil(
                        k * w
                    )

                    if wCal > 0 and wCal <= IMG_SIZE:

                        imgResize = cv2.resize(
                            imgCrop,
                            (wCal, IMG_SIZE)
                        )

                        wGap = math.ceil(
                            (IMG_SIZE - wCal) / 2
                        )

                        imgWhite[
                            :,
                            wGap:wGap + wCal
                        ] = imgResize

                # -------------------------------------------------
                # HORIZONTAL
                # -------------------------------------------------

                else:

                    k = IMG_SIZE / w

                    hCal = math.ceil(
                        k * h
                    )

                    if hCal > 0 and hCal <= IMG_SIZE:

                        imgResize = cv2.resize(
                            imgCrop,
                            (IMG_SIZE, hCal)
                        )

                        hGap = math.ceil(
                            (IMG_SIZE - hCal) / 2
                        )

                        imgWhite[
                            hGap:hGap + hCal,
                            :
                        ] = imgResize

                # -------------------------------------------------
                # PREDICTION
                # -------------------------------------------------

                prediction, index = classifier.getPrediction(
                    imgWhite,
                    draw=False
                )

                if 0 <= index < len(LABELS):

                    label = LABELS[index]

                else:

                    label = "Unknown"

                # -------------------------------------------------
                # CORRECT SIGN
                # -------------------------------------------------

                if label == target:

                    combo += 1

                    max_combo = max(
                        max_combo,
                        combo
                    )

                    correct_time += delta

                    # Base points
                    points = (
                        POINTS_PER_SECOND
                        * delta
                    )

                    # Combo bonus
                    if combo > 20:

                        points += (
                            COMBO_BONUS
                            * delta
                        )

                    round_score += points

                    box_color = (
                        80,
                        220,
                        100
                    )

                    status = "CORRECT!"

                # -------------------------------------------------
                # WRONG SIGN
                # -------------------------------------------------

                else:

                    combo = 0

                    box_color = (
                        80,
                        80,
                        220
                    )

                    status = "WRONG SIGN"

                # -------------------------------------------------
                # HAND BOX
                # -------------------------------------------------

                cv2.rectangle(
                    imgOutput,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    4
                )

                cv2.putText(
                    imgOutput,
                    label,
                    (x, max(50, y - 20)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.8,
                    box_color,
                    3
                )

                # -------------------------------------------------
                # STATUS
                # -------------------------------------------------

                cv2.putText(
                    imgOutput,
                    status,
                    (700, 550),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.3,
                    box_color,
                    3
                )

                # -------------------------------------------------
                # COMBO
                # -------------------------------------------------

                if combo > 20:

                    cv2.putText(
                        imgOutput,
                        f"COMBO x{combo // 20 + 1}",
                        (700, 610),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 200, 255),
                        3
                    )

        else:

            combo = 0

            cv2.putText(
                imgOutput,
                "SHOW YOUR HAND",
                (700, 550),
                cv2.FONT_HERSHEY_COMPLEX,
                1.2,
                (200, 200, 200),
                3
            )

        # -------------------------------------------------
        # PROGRESS BAR
        # -------------------------------------------------

        bar_x = 30
        bar_y = 620
        bar_width = 550
        bar_height = 25

        progress = remaining / ROUND_TIME

        cv2.rectangle(
            imgOutput,
            (bar_x, bar_y),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (60, 60, 70),
            -1
        )

        cv2.rectangle(
            imgOutput,
            (bar_x, bar_y),
            (
                bar_x + int(bar_width * progress),
                bar_y + bar_height
            ),
            (100, 220, 255),
            -1
        )

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        cv2.imshow(
            "Sign Challenge",
            imgOutput
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            return total_score, False


# =========================================================
# RESULT SCREEN
# =========================================================

def result_screen(
    player_name,
    score
):

    save_player(
        player_name,
        round(score)
    )

    while True:

        success, img = cap.read()

        if not success:
            return False

        img = cv2.flip(
            img,
            1
        )

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (100, 70),
            (1180, 650),
            (20, 20, 30),
            -1
        )

        img = cv2.addWeighted(
            overlay,
            0.9,
            img,
            0.1,
            0
        )

        draw_center_text(
            img,
            "GAME OVER",
            160,
            2,
            (255, 255, 255),
            4
        )

        draw_center_text(
            img,
            player_name,
            240,
            1.3,
            (180, 180, 180),
            2
        )

        draw_center_text(
            img,
            f"{round(score)} POINTS",
            350,
            2.5,
            (100, 220, 100),
            4
        )

        draw_center_text(
            img,
            "LEADERBOARD",
            430,
            1.2,
            (255, 220, 100),
            3
        )

        records = get_top_players()

        y = 480

        for i, player in enumerate(records):

            text = (
                f"{i + 1}. "
                f"{player['name']} "
                f"- {player['score']} pts"
            )

            cv2.putText(
                img,
                text,
                (320, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (230, 230, 230),
                2
            )

            y += 32

        cv2.putText(
            img,
            "ENTER = Next Player",
            (380, 630),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 220, 100),
            2
        )

        cv2.imshow(
            "Sign Challenge",
            img
        )

        key = cv2.waitKey(30) & 0xFF

        if key == 13:

            return True

        if key == ord("q"):

            return False


# =========================================================
# MAIN GAME
# =========================================================

def main():

    while True:

        # -------------------------------------------------
        # PLAYER NAME
        # -------------------------------------------------

        player_name = welcome_screen()

        if player_name is None:
            break

        total_score = 0

        # -------------------------------------------------
        # ROUNDS
        # -------------------------------------------------

        for round_number in range(
            1,
            TOTAL_ROUNDS + 1
        ):

            print(
                f"Starting round {round_number}"
            )

            # Countdown

            if not countdown():
                cap.release()
                cv2.destroyAllWindows()
                return

            # Game round

            total_score, running = play_round(
                player_name,
                round_number,
                total_score
            )

            if not running:

                cap.release()
                cv2.destroyAllWindows()
                return

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        next_player = result_screen(
            player_name,
            total_score
        )

        if not next_player:
            break

    cap.release()

    cv2.destroyAllWindows()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()