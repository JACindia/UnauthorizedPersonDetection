#!/usr/bin/env python3
"""
Employee Face Verification with InsightFace (GPU) + Telegram Alerts
Single asyncio event loop with flood-control-safe Telegram sending.
"""

import asyncio
from pathlib import Path
from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from telegram import Bot
from telegram.error import RetryAfter, TelegramError

# ===================== CONFIG ===================== #
EMPLOYEE_FOLDER = "employees"
NEW_IMAGES_FOLDER = "new_images"

# Replace with your actual token
TELEGRAM_TOKEN = "8548372776:AAEgz4mOaDBU8HyrCklRHFHsnxT9U7HtRA0"
TELEGRAM_CHAT_ID = "-5025882106"

SIMILARITY_THRESHOLD = 0.60
EMP_DET_SCORE_TH = 0.30
NEW_DET_SCORE_TH = 0.20

MODEL_NAME = "buffalo_l"
CTX_ID = 0
DET_SIZE = (640, 640)

# Basic throttle: Telegram recommends <= 1 message/second per chat
# https://core.telegram.org/bots/faq
SEND_DELAY_SECONDS = 1.1


# ===================== CORE VERIFIER ===================== #

class FaceVerifier:
    def __init__(self):
        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CUDAExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)

    def _get_faces(self, img: np.ndarray):
        return self.app.get(img, max_num=5)

    def get_embeddings(self, image_path: str, det_th: float) -> List[np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            return []

        faces = self._get_faces(img)
        embs = [f.embedding for f in faces if getattr(f, "det_score", 1.0) >= det_th]
        if embs:
            return embs

        # Try a slightly scaled-up image if no face was detected
        h, w = img.shape[:2]
        scale = 1.3
        resized = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        faces2 = self._get_faces(resized)
        embs2 = [f.embedding for f in faces2 if getattr(f, "det_score", 1.0) >= det_th]
        return embs2

    @staticmethod
    def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    def is_employee(self, new_embs: List[np.ndarray], employee_embs: List[np.ndarray]) -> bool:
        if not employee_embs or not new_embs:
            return False
        for ne in new_embs:
            sims = [self.cosine_similarity(ne, ee) for ee in employee_embs]
            if sims and max(sims) >= SIMILARITY_THRESHOLD:
                return True
        return False


# ===================== ASYNC TELEGRAM HELPERS ===================== #

async def tg_text(bot: Bot, msg: str):
    """
    Send a text message with flood-control handling.
    Retries on RetryAfter and throttles to ~1 msg/sec.
    """
    while True:
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            # basic per-message delay to reduce chance of hitting rate limit
            await asyncio.sleep(SEND_DELAY_SECONDS)
            break
        except RetryAfter as e:
            wait_for = int(getattr(e, "retry_after", 5)) + 1
            print(f"[Telegram] Flood control hit for send_message. Sleeping {wait_for} s.")
            await asyncio.sleep(wait_for)
        except TelegramError as e:
            # Log and stop trying this message
            print(f"[Telegram] TelegramError while sending message: {e}")
            break
        except Exception as e:
            print(f"[Telegram] Unexpected error while sending message: {e}")
            break


async def tg_photo(bot: Bot, path: str, caption: str):
    """
    Send a photo with caption with flood-control handling.
    Retries on RetryAfter and throttles to ~1 msg/sec.
    """
    while True:
        try:
            with open(path, "rb") as f:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=f, caption=caption)
            await asyncio.sleep(SEND_DELAY_SECONDS)
            break
        except RetryAfter as e:
            wait_for = int(getattr(e, "retry_after", 5)) + 1
            print(f"[Telegram] Flood control hit for send_photo. Sleeping {wait_for} s.")
            await asyncio.sleep(wait_for)
        except TelegramError as e:
            print(f"[Telegram] TelegramError while sending photo: {e}")
            break
        except Exception as e:
            print(f"[Telegram] Unexpected error while sending photo: {e}")
            break


# ===================== ASYNC MAIN ===================== #

async def async_main():
    bot = Bot(token=TELEGRAM_TOKEN)

    emp_dir = Path(EMPLOYEE_FOLDER)
    new_dir = Path(NEW_IMAGES_FOLDER)

    if not emp_dir.exists():
        await tg_text(bot, f"Employee folder not found: {emp_dir.resolve()}")
        return
    if not new_dir.exists():
        await tg_text(bot, f"New images folder not found: {new_dir.resolve()}")
        return

    await tg_text(bot, "Initializing InsightFace on GPU...")
    verifier = FaceVerifier()
    await tg_text(bot, "InsightFace initialized (buffalo_l).")

    # ---- Load employee embeddings ----
    employee_embs: List[np.ndarray] = []
    emp_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        emp_files.extend(emp_dir.glob(ext))

    await tg_text(bot, f"Found {len(emp_files)} employee files.")

    for p in emp_files:
        embs = verifier.get_embeddings(str(p), det_th=EMP_DET_SCORE_TH)
        if embs:
            employee_embs.extend(embs)
        else:
            await tg_text(bot, f"No face detected in employee image: {p.name}")

    if not employee_embs:
        await tg_text(bot, "No employee faces loaded. Stopping.")
        return

    # ---- Process new images ----
    new_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        new_files.extend(new_dir.glob(ext))

    await tg_text(bot, f"Found {len(new_files)} new images to check.")
    if not new_files:
        return

    for np_path in new_files:
        np_path = Path(np_path)
        new_embs = verifier.get_embeddings(str(np_path), det_th=NEW_DET_SCORE_TH)

        if not new_embs:
            await tg_text(bot, f"No face detected in new image: {np_path.name}")
            continue

        if not verifier.is_employee(new_embs, employee_embs):
            caption = f"UNAUTHORIZED PERSON DETECTED\nFile: {np_path.name}"
            await tg_photo(bot, str(np_path), caption)
        else:
            await tg_text(bot, f"Authorized employee detected in: {np_path.name}")

    await tg_text(bot, "Verification run completed.")


# ===================== ENTRYPOINT ===================== #

if __name__ == "__main__":
    asyncio.run(async_main())
