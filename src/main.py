import os
import io
import json
import torch
import base64
import re
import pandas as pd
import numpy as np
import difflib
import asyncio
import sqlite3
import uuid
import aiosqlite
import logging
import time
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from fastapi.staticfiles import StaticFiles

# настройки
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REDIRECT_URI = 'http://127.0.0.1:8000/callback'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'music.db')
RESOURCE_LOG_FILE = "resource_usage.csv"

app = FastAPI()
analysis_tasks = {} 

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
logger = logging.getLogger("MusicAI")


def load_model(quantization_method: str = "nf4"):
    print(f"Loading {MODEL_ID} with {quantization_method} quantization...")
    
    if quantization_method == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("Используется: 4-bit NF4 + Double Quant")
        
    elif quantization_method == "fp8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Используется: 8-bit")
        
    elif quantization_method == "none":
        bnb_config = None
        print("Загрузка без квантизации (FP16/BF16)")
        
    else:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        trust_remote_code=True
    ).eval()
    
    print(f"Модель успешно загружена ({quantization_method})")
    print(f"VRAM allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n")
    
    return model, processor

# инициализация
@asynccontextmanager
async def lifespan(app: FastAPI):
    model, processor = load_model(quantization_method="nf4")
    
    app.state.model = model
    app.state.processor = processor
    app.state.quant_method = "nf4"
    
    # ---------------------------------------------------------
    # Фоновая задача: TTL очистка словаря analysis_tasks
    # ---------------------------------------------------------
    async def clean_expired_tasks():
        TTL_SECONDS = 3600  # Время жизни задачи (1 час)
        while True:
            await asyncio.sleep(600)  # Просыпаемся каждые 10 минут
            current_time = time.time()
            # Находим задачи старше TTL
            expired_keys = [
                t_id for t_id, task_data in analysis_tasks.items()
                if current_time - task_data.get("created_at", current_time) > TTL_SECONDS
            ]
            for t_id in expired_keys:
                del analysis_tasks[t_id]
                logger.info(f"Task {t_id} removed from memory by TTL cleanup.")

    # Запускаем таску параллельно с сервером
    cleanup_task = asyncio.create_task(clean_expired_tasks())
    
    yield
    
    # При выключении сервера гасим сборщик мусора и чистим GPU
    cleanup_task.cancel()
    print("Clearing resources...")
    if hasattr(app.state, 'model'):
        del app.state.model
    if hasattr(app.state, 'processor'):
        del app.state.processor
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

analysis_tasks = {} 
model_lock = asyncio.Lock()

if not os.path.exists(RESOURCE_LOG_FILE):
    with open(RESOURCE_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("timestamp,task_id,image_size_kb,cpu_start_%,cpu_end_%,ram_start_%,ram_end_%,peak_vram_gb,peak_power_w\n")

def get_music_dataframe():
    """Безопасная загрузка данных из SQL с жесткой валидацией структуры"""
    try:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Файл БД {DB_PATH} не найден.")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'")
        if not cursor.fetchone():
            raise RuntimeError("Таблица 'tracks' отсутствует. Выполните в терминале: sqlite3 src/music.db < db/schema.sql")

        df = pd.read_sql("SELECT * FROM tracks", conn)
        conn.close()
        
        # Проверяем, есть ли данные в таблице
        if df.empty:
            raise RuntimeError("Таблица 'tracks' пуста. Пожалуйста, наполните базу данных треками перед запуском.")
            
        target_cols = ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'popularity']
        
        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.5)
        
        return df
        
    except Exception as e:

        print(f"\n{'='*50}")
        print(f" КРИТИЧЕСКАЯ ОШИБКА ЗАПУСКА:")
        print(f"{e}")
        print(f"{'='*50}\n")
        exit(1)

def init_feedback_db():
    """Создание таблицы для хранения оценок пользователей"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            rating INTEGER,
            valence REAL,
            energy REAL,
            danceability REAL,
            acousticness REAL,
            instrumentalness REAL,
            hue REAL,
            saturation REAL,
            brightness REAL,
            description TEXT,
            tracks_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

#init_feedback_db()

def migrate_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE ratings ADD COLUMN tracks_json TEXT")
        conn.commit()
        print("Колонка tracks_json успешно добавлена.")
    except sqlite3.OperationalError:
        print("Колонка tracks_json уже существует.")
    finally:
        conn.close()

init_feedback_db()
migrate_db()

music_df = get_music_dataframe()
if not music_df.empty:
    music_df = music_df.reset_index(drop=True)
    print(f"Успех! Загружено {len(music_df)} треков из базы данных.")

ALL_UNIQUE_GENRES = music_df['genre'].dropna().unique().tolist()
GENRE_MATCH_CACHE = {}

top_genres = music_df['genre'].value_counts().head(250).index.tolist()
genres_for_prompt = ", ".join(top_genres)

VLM_PROMPT = f"""
You are an Artificial Intelligence acting as a Creative Music Curator and Aesthetic Matchmaking Expert. 
Your task is to conduct an in-depth, contextual analysis of the image provided in order to select music compositions that perfectly match the visual atmosphere by outputting psycho-acoustic features.

AVAILABLE GENRES:
{genres_for_prompt}

1. ANALYSIS REQUIREMENTS (Human Imitation):
- Style and genre: Select 1-3 genres ONLY from the AVAILABLE GENRES list above that best match the image.
- Emotional Tone (Valence): Assess the overall mood (how positive/happy or sad/tense it is). Range: 0.0 (most negative/sad) to 1.0 (most positive/happy).
- Energy: Rate the dynamics, intensity, and complexity. Range: 0.0 (calm, ambient, background) to 1.0 (high-energy, aggressive, loud).
- Rhythm/Danceability: Rate the tempo and rhythm that matches the image. Range: 0.0 (slow, no obvious beat) to 1.0 (fast, danceable, with a clear rhythm).
- Acousticness: Rate how much the resulting music should be acoustic (un-amplified instruments) vs. electronic/synthesized. Range: 0.0 (electronic/processed) to 1.0 (purely acoustic).
- Instrumentalness: Rate the likelihood of the track being instrumental (no vocals). Range: 0.0 (high vocals) to 1.0 (pure instrumental).
- Description: Provide a short, captivating description of the image's aesthetic and mood.

2. OUTPUT FORMAT (JSON Schema):
Your response must contain ONLY one single, perfectly structured JSON object. Do not include ANY text, explanations, or characters outside of the JSON block. You MUST include all keys listed below.

{{
  "vlm_valence": 0.0,
  "vlm_energy": 0.0,
  "vlm_danceability": 0.0,
  "vlm_acousticness": 0.0,
  "vlm_instrumentalness": 0.0,
  "seed_genre_query": "string",
  "vlm_description": "string"
}}
"""

def get_dominant_colors(image_pil, num_colors=5):
    """Извлекает доминирующие цвета с помощью Pillow"""
    # даунскейлим картинку для скорости
    img = image_pil.copy()
    img.thumbnail((150, 150))
    # конвертим и получаем палитру ргб
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    
    colors = []
    for i in range(min(num_colors, len(color_counts))):
        palette_index = color_counts[i][1]
        r = palette[palette_index*3]
        g = palette[palette_index*3+1]
        b = palette[palette_index*3+2]
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors

def get_image_metrics(image_pil: Image):
    """
    Извлекает среднюю насыщенность, яркость и цветовой тон (Hue).
    Возвращает значения в диапазоне [0.0, 1.0].
    """
    hsv_img = image_pil.convert('HSV')
    # Разделяем каналы
    h_channel = np.array(hsv_img.getchannel('H')) / 255.0 # Hue (Тон)
    s_channel = np.array(hsv_img.getchannel('S')) / 255.0 # Saturation (Насыщенность)
    v_channel = np.array(hsv_img.getchannel('V')) / 255.0 # Value (Яркость)
    
    avg_hue = float(np.mean(h_channel))
    avg_saturation = float(np.mean(s_channel))
    avg_brightness = float(np.mean(v_channel))
    
    return avg_hue, avg_saturation, avg_brightness

if torch.cuda.is_available():
    device_map = "auto"
    print(f"GPU найдена: {torch.cuda.get_device_name(0)}")
else:
    device_map = "cpu"
    print("ВНИМАНИЕ: GPU не найдена, модель будет работать крайне медленно!")

def get_vlm_analysis_local(image_pil: Image, model, processor):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        analysis_image = image_pil.copy()
        analysis_image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a JSON-only generator. Never provide prose. Output ends immediately after the closing brace '}'."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": analysis_image},
                    {"type": "text", "text": VLM_PROMPT}
                ]
            },
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text], 
            images=[analysis_image], 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=False, 
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        output_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        md_start = "```" + "json"
        md_end = "```"
        
        processed_text = output_text.replace(md_start, "").replace(md_end, "").strip()
        
        logger.info(f"Model response received. Length: {len(processed_text)}")

        try:
            match = re.search(r'(\{.*\})', processed_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            
            return json.loads(processed_text)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON Parsing failed: {e}. Trying simple recovery...")
            
            if "{" in processed_text and "}" not in processed_text:
                try:
                    return json.loads(processed_text + "}")
                except:
                    pass
            
            return {
                "vlm_valence": 0.5, 
                "vlm_energy": 0.5, 
                "vlm_danceability": 0.5,
                "vlm_acousticness": 0.5, 
                "vlm_instrumentalness": 0.5,
                "seed_genre_query": "ambient",
                "vlm_description": "Ошибка обработки формата JSON."
            }

    except Exception as e:
        logger.error(f"Inference process crashed: {e}", exc_info=True)
        return {
            "vlm_valence": 0.5, 
            "vlm_energy": 0.5, 
            "vlm_description": "Критическая ошибка инференса."
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def benchmark_inference(model, processor, image_pil, runs=3):
    times = []
    vram_usage = []
    
    for i in range(runs):
        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        result = get_vlm_analysis_local(image_pil, model, processor)
        
        end = time.perf_counter()
        times.append(end - start)
        
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            vram_usage.append(peak)
        
        print(f"Запуск {i+1}: {end-start:.2f} сек | Peak VRAM: {peak:.2f} GB")
    
    print(f"\nСреднее время: {np.mean(times):.2f} ± {np.std(times):.2f} сек")
    print(f"Средний Peak VRAM: {np.mean(vram_usage):.2f} GB")
    return result
            
async def run_analysis(task_id: str, image_data: bytes, model, processor):
    start_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu_start = psutil.cpu_percent(interval=None)
    ram_start = psutil.virtual_memory().percent
    try:
        logger.info(f"Task {task_id}: Starting analysis")
        current_loop = asyncio.get_running_loop()
        
        with Image.open(io.BytesIO(image_data)) as img:
            original_image = img.convert("RGB")
            image_size_kb = len(image_data) / 1024
            
            buffered = io.BytesIO()
            original_image.save(buffered, format="JPEG", quality=85)
            image_data_url = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            
            async with model_lock:
                raw_analysis = await current_loop.run_in_executor(
                    None, 
                    get_vlm_analysis_local, 
                    original_image,
                    model, 
                    processor 
                )
            
            palette_hex = await current_loop.run_in_executor(None, get_dominant_colors, original_image)
            avg_hue, avg_sat, avg_bright = await current_loop.run_in_executor(None, get_image_metrics, original_image)

        original_description = raw_analysis.get("vlm_description", "No description")
        try:
            translator = GoogleTranslator(source='en', target='ru')
            russian_description = await current_loop.run_in_executor(
                None, translator.translate, original_description
            )
        except Exception as te:
            logger.warning(f"Translation failed: {te}")
            russian_description = original_description

        weight_vlm, weight_vis = 0.7, 0.3

        vlm_valence = float(raw_analysis.get("vlm_valence", 0.5))
        vlm_energy = float(raw_analysis.get("vlm_energy", 0.5))
        vlm_acousticness = float(raw_analysis.get("vlm_acousticness", 0.5))
        vlm_danceability = float(raw_analysis.get("vlm_danceability", 0.5))

        hue_bias = -0.1 if 0.45 < avg_hue < 0.75 else 0.05 if (avg_hue < 0.2 or avg_hue > 0.8) else 0
        
        combined_valence = (vlm_valence * weight_vlm) + (avg_bright * weight_vis) + hue_bias
        combined_energy = (vlm_energy * weight_vlm) + (avg_sat * weight_vis)
        
        is_warm = 1.0 if (avg_hue < 0.2 or avg_hue > 0.8) else 0.0
        combined_acousticness = (vlm_acousticness * weight_vlm) + (is_warm * weight_vis)
        combined_danceability = (vlm_danceability * weight_vlm) + (avg_sat * weight_vis)

        analysis = {
            "mood": round(max(0, min(1, combined_valence)), 3),
            "energy": round(max(0, min(1, combined_energy)), 3),
            "danceability": round(max(0, min(1, combined_danceability)), 3),
            "acousticness": round(max(0, min(1, combined_acousticness)), 3),
            "instrumentalness": round(float(raw_analysis.get("vlm_instrumentalness", 0.5)), 3),
            "description": russian_description,
            "hue": round(avg_hue, 3),
            "saturation": round(avg_sat, 3),
            "brightness": round(avg_bright, 3),
            "hue_degrees": int(avg_hue * 360),
            "saturation_pct": int(avg_sat * 100),
            "brightness_pct": int(avg_bright * 100)
        }
        
        target_vector = np.array([[
            analysis['mood'], 
            analysis['energy'], 
            analysis['danceability'],
            analysis['acousticness'], 
            analysis['instrumentalness']
        ]], dtype=float)
        
        cols = ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness']
        features = music_df[cols].values.astype(float)
        
        cos_sim = cosine_similarity(target_vector, features)[0]
        dist = np.linalg.norm(features - target_vector, axis=1)
        euclidean_score = 1 / (1 + dist)
        hybrid_similarities = (cos_sim * 0.7) + (euclidean_score * 0.3)
       
        genre_boost = np.zeros(len(music_df))
        seed_genres = raw_analysis.get("seed_genre_query", "").lower()
        
        if seed_genres:
            ai_tags = [t.strip() for t in seed_genres.replace(',', ' ').split() if len(t.strip()) > 2]
            if ai_tags:
                matched_genres = []
                for tag in ai_tags:
                    matches = difflib.get_close_matches(tag, ALL_UNIQUE_GENRES, n=2, cutoff=0.7)
                    matched_genres.extend(matches)
                
                if matched_genres:
                    mask = music_df['genre'].fillna('').isin(list(set(matched_genres)))
                    genre_boost = mask.astype(float)

        popularity_scores = music_df['popularity'].fillna(0).values.astype(float) / 100.0
        combined_scores = (hybrid_similarities * 0.75) + (genre_boost * 0.15) + (popularity_scores * 0.1)
        
        candidate_indices = np.argsort(combined_scores)[-80:] 
        unique_candidates = music_df.iloc[candidate_indices].copy()
        unique_candidates["similarity"] = combined_scores[candidate_indices]
        unique_candidates = unique_candidates.drop_duplicates(subset=['track_name', 'artist_name'])
      
        result_count = min(10, len(unique_candidates))
        if result_count > 0:
            top_tracks = unique_candidates.sample(n=result_count)
            top_tracks["similarity"] = np.clip(top_tracks["similarity"], 0.75, 0.98)
            top_tracks = top_tracks.sort_values(by="similarity", ascending=False)
            top_tracks_list = top_tracks.to_dict('records')
        else:
            top_tracks_list = []

        end_time = time.perf_counter()
        total_time = end_time - start_time

        analysis_tasks[task_id] = {
            "status": "completed",
            "analysis": analysis, 
            "tracks": top_tracks_list,
            "palette": palette_hex,
            "user_image": image_data_url,
            "processing_time": round(total_time, 2),
            "created_at": time.time()
        }
        
        cpu_end = psutil.cpu_percent(interval=None)
        ram_end = psutil.virtual_memory().percent
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        power_draw = 0
        if torch.cuda.is_available():
            try:
                import subprocess
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
                power_draw = float(output.decode('utf-8').strip())
            except:
                power_draw = 0
        
        with open("analysis_times.log", "a", encoding="utf-8") as f:
            f.write(f"{task_id},{total_time:.3f},{image_size_kb:.1f}KB\n")

        with open(RESOURCE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},{task_id},{image_size_kb:.1f},{cpu_start:.1f},{cpu_end:.1f},"
                    f"{ram_start:.1f},{ram_end:.1f},{peak_vram:.2f},{power_draw:.1f}\n")
            
            
        print(f"\n{'='*75}")
        print(f"ЗАВЕРШЕНО: Task {task_id}")
        print(f"Общее время анализа: {total_time:.2f} секунд")
        print(f"CPU: {cpu_start:.1f}% → {cpu_end:.1f}%")
        print(f"RAM: {ram_start:.1f}% → {ram_end:.1f}%")
        print(f"Peak VRAM: {peak_vram:.2f} ГБ")
        print(f"Размер изображения: {image_size_kb:.1f} КБ")
        print(f"Power Consumption : {power_draw:.1f} Вт")
        print(f"{'='*75}\n")
        
        
        analysis_tasks[task_id] = {
            "status": "completed",
            "analysis": analysis, 
            "tracks": top_tracks_list,
            "palette": palette_hex,
            "user_image": image_data_url,
            "processing_time": round(total_time, 2)   # добавляем в результат
        }        
        
        logger.info(f"Task {task_id}: Analysis completed successfully")

    except Exception as e:
        logger.exception(f"Task {task_id}: Critical background error")
        analysis_tasks[task_id] = {
            "status": "error", 
            "message": f"Ошибка обработки: {str(e)}"
        }
        
        
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(
    request: Request, 
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            logger.warning(f"Rejected file with type: {file.content_type}")
            return HTMLResponse("Ошибка: Поддерживаются только форматы JPEG и PNG", status_code=400)

        MAX_SIZE = 10 * 1024 * 1024  # 10MB
        img_bytes = await file.read()
        if len(img_bytes) > MAX_SIZE:
            return HTMLResponse("Ошибка: Файл слишком большой (макс. 10 МБ)", status_code=400)

        task_id = str(uuid.uuid4())

        analysis_tasks[task_id] = {
            "status": "processing",
            "created_at": time.time()
        }

        model = request.app.state.model
        processor = request.app.state.processor
   
        background_tasks.add_task(
            run_analysis, 
            task_id, 
            img_bytes, 
            model, 
            processor
        )
        
        logger.info(f"Task created: {task_id}. Image size: {len(img_bytes)} bytes")

        return templates.TemplateResponse("loading.html", {
            "request": request,
            "task_id": task_id
        })

    except Exception as e:
        logger.exception(f"Critical upload error: {e}")
        return HTMLResponse(
            f"Системная ошибка при загрузке: {str(e)}", 
            status_code=500
        )

@app.get("/result/{task_id}", response_class=HTMLResponse)
async def get_result(request: Request, task_id: str):
    task = analysis_tasks.get(task_id)
    if not task or task.get("status") != "completed":
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "analysis": task["analysis"],
        "tracks": task["tracks"],
        "palette": task["palette"],
        "user_image": task["user_image"],
        "brightness": task["analysis"].get("brightness_pct", 0),
        "saturation": task["analysis"].get("saturation_pct", 0),
        "hue": task["analysis"].get("hue_degrees", 0)
    })

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = analysis_tasks.get(task_id)
    if not task:
        return {"status": "not_found"}
    
    response = {"status": task.get("status")}
    if task.get("status") == "error":
        response["message"] = task.get("message", "Unknown error during analysis")
    
    return response

executor = ThreadPoolExecutor(max_workers=4)

def generate_image_sync(tracks, palette, user_image_base64):
    """Синхронная логика отрисовки, вынесенная из основного потока"""
    img_data = base64.b64decode(user_image_base64)
    
    with Image.open(io.BytesIO(img_data)).convert("RGB") as base_img:
        canvas_w, canvas_h = 1080, 1080
        resized_img = base_img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        background = resized_img.filter(ImageFilter.GaussianBlur(radius=50))
        
        overlay = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 160))
        background.paste(overlay, (0, 0), overlay)
        
        draw = ImageDraw.Draw(background, "RGBA")
        
        def get_font(size):
            try: return ImageFont.truetype("arial.ttf", size)
            except: return ImageFont.load_default()

        start_y = 100
        draw.rectangle([60, start_y - 20, 700, start_y + 190], fill=(255, 255, 255, 40))
        draw.text((80, start_y), "ЦВЕТОВАЯ ПАЛИТРА", fill=(255, 255, 255), font=get_font(24))
        
        for i, hex_color in enumerate(palette[:5]):
            x, y = 80 + (i * 120), start_y + 50
            draw.ellipse([x - 5, y - 5, x + 85, y + 85], fill=(0, 0, 0, 255))
            draw.ellipse([x, y, x + 80, y + 80], fill=hex_color)
            draw.text((x + 5, y + 95), hex_color.upper(), fill=(255, 255, 255), font=get_font(18))

        draw.text((80, 320), "ВАШ ВИЗУАЛЬНЫЙ ПЛЕЙЛИСТ", fill="#1DB954", font=get_font(54))
        
        y_offset = 450
        for i, track in enumerate(tracks[:10]):
            text = f"{i+1}. {track['track_name']} — {track['artist_name']}"
            display_text = text[:57] + "..." if len(text) > 60 else text
            draw.text((80, y_offset), display_text, fill=(255, 255, 255), font=get_font(28))
            
            # Сходство (Similarity)
            sim_text = f"{int(track.get('similarity', 0)*100)}%"
            draw.text((920, y_offset + 2), sim_text, fill=(255, 255, 255, 180), font=get_font(24))
            y_offset += 55

        draw.text((430, 1030), "Система подбора музыки по изображению", fill=(255, 255, 255, 80), font=get_font(24))

        img_byte_arr = io.BytesIO()
        background.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        return img_byte_arr

@app.post("/download_card")
async def download_card(request: Request):
    try:
        form_data = await request.form()
        tracks = json.loads(form_data.get("tracks_data", "[]"))
        palette = json.loads(form_data.get("palette_data", "[]"))
        image_field = form_data.get("user_image", "")
        
        if "," not in image_field:
            return HTMLResponse("Invalid image data", status_code=400)
            
        user_image_base64 = image_field.split(",")[1]

        # Запускаем тяжелую графику в отдельном потоке
        loop = asyncio.get_event_loop()
        img_byte_arr = await loop.run_in_executor(
            executor, 
            generate_image_sync, 
            tracks, palette, user_image_base64
        )

        return StreamingResponse(
            img_byte_arr, 
            media_type="image/jpeg", 
            headers={"Content-Disposition": "attachment; filename=music_card.jpg"}
        )
    except Exception as e:
        logger.error(f"Card Generation Error: {e}")
        return HTMLResponse("Error generating image card", status_code=500)

@app.post("/download_csv")
async def download_csv(request: Request):
    form_data = await request.form()
    tracks_json = form_data.get("tracks_data")
    
    if not tracks_json:
        return RedirectResponse("/", status_code=303)
        
    try:
        tracks = json.loads(tracks_json)
        df = pd.DataFrame(tracks).fillna('') 

        if not df.empty:
            df = df.drop_duplicates(subset=['track_name', 'artist_name'])
            
            cols = [c for c in ['track_name', 'artist_name'] if c in df.columns]
            df = df[cols]
            df.columns = ['Название', 'Исполнитель']

            stream = io.StringIO()
            df.to_csv(stream, index=False, encoding='utf-8')
            
            content = "\ufeff" + stream.getvalue()
            
            return StreamingResponse(
                iter([content]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=playlist.csv",
                    "Content-Type": "text/csv; charset=utf-8-sig"
                }
            )
    except Exception as e:
        logger.error(f"CSV Export Error: {e}")
        return RedirectResponse("/", status_code=303)

@app.post("/rate")
async def submit_rating(request: Request):
    try:
        data = await request.json()
        rating = int(data.get("rating", 0))
        analysis = data.get("analysis_data", {})
        all_tracks = data.get("tracks", [])
        
        if not (1 <= rating <= 5):
            return {"status": "error", "message": "Invalid rating"}

        allowed_keys = {'track_id', 'track_name', 'artist_name', 'genre', 'popularity', 'similarity'}
        clean_tracks = [{k: v for k, v in t.items() if k in allowed_keys} for t in all_tracks]
        tracks_as_json = json.dumps(clean_tracks, ensure_ascii=False)

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
                INSERT INTO ratings (
                    rating, valence, energy, danceability, 
                    acousticness, instrumentalness, hue, 
                    saturation, brightness, description,
                    tracks_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rating,
                analysis.get("mood"),
                analysis.get("energy"),
                analysis.get("danceability"),
                analysis.get("acousticness"),
                analysis.get("instrumentalness"),
                analysis.get("hue"),
                analysis.get("saturation"),
                analysis.get("brightness"),
                analysis.get("description"),
                tracks_as_json
            ))
            await db.commit()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Rating Save Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn

    benchmark_mode = False   # тут менять

    if benchmark_mode:
        print("\n" + "="*70)
        print("=== ЗАПУСК БЕНЧМАРКА КВАНТИЗАЦИИ ===")
        print("="*70)

        model, processor = load_model(quantization_method="nf4")  # ← меняй метод здесь
        test_image_path = "levitan.jpg"
        
        try:
            test_image = Image.open(test_image_path).convert("RGB")
            print(f"Тестовое изображение загружено: {test_image.size}")
            benchmark_inference(model, processor, test_image, runs=5)
        except FileNotFoundError:
            print(f"Файл {test_image_path} не найден!")
        except Exception as e:
            print(f"Ошибка: {e}")

        del model, processor
        torch.cuda.empty_cache()
        exit()

    print("Запуск веб-сервера...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
