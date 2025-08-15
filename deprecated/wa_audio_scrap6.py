#!/usr/bin/env python3
# wa_voicenotes_capture_plus_final.py
#
# Versión final integrada:
# - descarga via menú "Descargar" o captura blobs (hook createObjectURL)
# - evita duplicados por hash (sha256)
# - llama automáticamente a convert_ogg_to_wav.py (opcional)
# - helpers para crear plantillas (selector screenshot + recorte manual)
# - reintentos, grid visual, screenshots para depuración
#
# Requisitos:
# pip install selenium webdriver-manager opencv-python pillow numpy
# Tener convert_ogg_to_wav.py en la misma carpeta (o ajustar ruta)
#
# USO: editar CONFIG abajo y ejecutar:
# python wa_voicenotes_capture_plus_final.py
#
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import functools
from pathlib import Path
from typing import List, Tuple, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Vision libs
import cv2
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
BASE_SAVE_DIR = Path("wa_voicenotes")
DOWNLOAD_DIR = Path("wa_downloads")     # Chrome download dir when using "Descargar" menu
DECODED_JS_DIR = Path("decoded_inline_js")
TEMPLATES_DIR = Path("templates")
SCREENSHOTS_DIR = Path("screenshots")
TMP_SCREENSHOT = Path("tmp_whatsapp_full.png")

CHATS_TO_SCAN = 20
WAIT_TIMEOUT = 90          # tiempo para esperar que cargue lista de chats
CLICK_WAIT = 0.6           # tiempo tras clicks para permitir UI/JS
MAX_RETRIES = 4
RETRY_BACKOFF = 1.7

# Persistencia de perfil Chrome (evita QR cada vez)
CHROME_USER_DATA_DIR = r"C:\Users\soyko\AppData\Local\Google\Chrome\User Data\wa_profile"
CHROME_PROFILE = None  # "Profile 1" si usas perfiles

# Opciones de comportamiento
CLOSE_BROWSER_AT_END = False
SEND_LOVABLES = False
CALL_CONVERTER = True            # si True llama convert_ogg_to_wav.py tras guardar un .ogg
CONVERTER_SCRIPT = "convert_ogg_to_wav.py"
CONVERTER_ARGS = ["--sr", "22050", "--channels", "1", "--jobs", "2", "--overwrite"]  # --overwrite opcional

# Mensajes cariñosos (opcional)
LOVABLE_CONTACTS = {
    "AA Mama": "Hola mamá ❤️ Te quiero mucho. Un abrazo grande de parte mía.",
    "Jorge": "Este es un mensaje automático ",
    "JavBitch": "¡Ey, crack! Abrazo, gracias por todo 😄",
    "Mi ex mujer ♥️🪀": "Solo quería mandarte un saludo cariñoso. 😉"
}

# -----------------------------------------
for d in (BASE_SAVE_DIR, DOWNLOAD_DIR, DECODED_JS_DIR, TEMPLATES_DIR, SCREENSHOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

downloaded_index_file = BASE_SAVE_DIR / "downloaded.json"
if downloaded_index_file.exists():
    try:
        with open(downloaded_index_file, "r", encoding="utf-8") as f:
            downloaded_index = json.load(f)
    except Exception:
        downloaded_index = {}
else:
    downloaded_index = {}

# downloaded_index structure:
# key = sha256 hex -> { "chat":..., "sender":..., "stored_path":..., "orig_filename":..., "timestamp":... }
# We will also store mapping chat|blobkey -> sha256 if available (not mandatory)

# ------------- Utilities --------------
def retry(times=3, delay=0.6, backoff=2.0, exceptions=(Exception,)):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(1, times + 1):
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    time.sleep(_delay)
                    _delay *= backoff
            return None
        return wrapper
    return deco

def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r'[\\/*?:"<>|]', "_", s)
    s = re.sub(r'\s+', " ", s)
    return s[:200]

def atomic_save_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def save_index():
    atomic_save_json(downloaded_index_file, downloaded_index)

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# ------------- Visual helpers (OpenCV) --------------
def find_template_positions(screenshot_path: Path, template_path: Path, threshold: float = 0.77) -> List[Tuple[int,int]]:
    img = cv2.imread(str(screenshot_path), cv2.IMREAD_UNCHANGED)
    tpl = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if img is None or tpl is None:
        return []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    positions = []
    h, w = tpl_gray.shape
    for pt in zip(*loc[::-1]):
        positions.append((int(pt[0] + w / 2), int(pt[1] + h / 2)))
    # filter close duplicates
    filtered = []
    for p in positions:
        if not any((abs(p[0]-q[0])<10 and abs(p[1]-q[1])<10) for q in filtered):
            filtered.append(p)
    return filtered

def take_panel_screenshot(driver, bbox: Tuple[int,int,int,int], out_path: Path) -> Path:
    x, y, w, h = bbox
    driver.save_screenshot(str(TMP_SCREENSHOT))
    img = Image.open(TMP_SCREENSHOT)
    cropped = img.crop((x, y, x + w, y + h))
    cropped.save(out_path)
    return out_path

# ------------- DOM / UI helpers --------------
def init_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--start-maximized")
    prefs = {
        "download.default_directory": str(DOWNLOAD_DIR.resolve()),
        "download.prompt_for_download": False,
        "profile.default_content_setting_values.automatic_downloads": 1,
        "safebrowsing.enabled": True
    }
    opts.add_experimental_option("prefs", prefs)
    if CHROME_USER_DATA_DIR:
        opts.add_argument(f"--user-data-dir={CHROME_USER_DATA_DIR}")
        if CHROME_PROFILE:
            opts.add_argument(f"--profile-directory={CHROME_PROFILE}")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    return driver

# Hook createObjectURL (captura blobs como dataURLs)
CREATE_OBJECT_URL_HOOK = r"""
(function(){
  if(window.__WA_BLOB_HOOKED) return;
  window.__WA_BLOB_HOOKED = true;
  window.__WA_BLOB_CACHE = window.__WA_BLOB_CACHE || {};
  const orig_create = URL.createObjectURL;
  URL.createObjectURL = function(obj) {
    try {
      if(obj && (obj instanceof Blob || obj instanceof File)) {
        const blobUrl = orig_create.call(URL, obj);
        try {
          const reader = new FileReader();
          reader.onload = function(e){
            try { window.__WA_BLOB_CACHE[blobUrl] = {dataUrl: e.target.result, createdAt: Date.now()}; }
            catch(err){ console.error("WA_BLOB_CACHE write err", err); }
          };
          reader.onerror = function(err){ console.error("WA FileReader err", err); };
          reader.readAsDataURL(obj);
        } catch(errInner) {
          window.__WA_BLOB_CACHE[blobUrl] = { dataUrl: null, createdAt: Date.now() };
        }
        return blobUrl;
      }
    } catch(e){ console.warn("Error hooking createObjectURL", e); }
    return orig_create.call(URL, obj);
  };
  window.__wa_get_blob_keys = function(){ return Object.keys(window.__WA_BLOB_CACHE || {}); };
  window.__wa_get_blob_entry = function(key){ return window.__WA_BLOB_CACHE && window.__WA_BLOB_CACHE[key] ? window.__WA_BLOB_CACHE[key] : null; };
  window.__wa_clear_blob_entry = function(key){ if(window.__WA_BLOB_CACHE && window.__WA_BLOB_CACHE[key]) delete window.__WA_BLOB_CACHE[key]; return true; };
})();
"""

def decode_inline_base64(page_source: str):
    pattern = re.compile(r'(?:src|href)="data:text/javascript;base64,([^"]+)"')
    found = pattern.findall(page_source)
    for i, b64 in enumerate(found, start=1):
        try:
            txt = base64.b64decode(b64).decode("utf-8", errors="replace")
            fname = DECODED_JS_DIR / f"inline_script_{i}.js"
            fname.write_text(txt, encoding="utf-8")
            print("[decoded inline js] saved:", fname)
        except Exception as e:
            print("[decoded inline js] error:", e)

def wait_for_login_and_pane(driver, timeout=WAIT_TIMEOUT):
    driver.get("https://web.whatsapp.com")
    print("Abriendo web.whatsapp.com — escanea el QR si es necesario.")
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#pane-side, div[role='grid'], div[role='region']"))
        )
    except Exception as e:
        Path("wa_page_source_debug.html").write_text(driver.page_source, encoding="utf-8")
        raise RuntimeError("No se detectó el panel lateral (#pane-side). Revisa wa_page_source_debug.html") from e
    start = time.time()
    while time.time() - start < timeout:
        elems = driver.find_elements(By.CSS_SELECTOR, "#pane-side [role='row'], #pane-side [role='button'], #pane-side [data-testid='cell-frame-container']")
        if elems and len(elems) > 0:
            print("Lista de chats detectada.")
            return
        time.sleep(1)
    Path("wa_page_source_debug.html").write_text(driver.page_source, encoding="utf-8")
    raise TimeoutError("La lista de chats no se cargó a tiempo. Revisa wa_page_source_debug.html")

def get_chat_elements(driver):
    selectors = [
        "#pane-side div[role='row']",
        "#pane-side div[role='button']",
        "#pane-side [data-testid='cell-frame-container']",
        "div[role='grid'] div[role='row']"
    ]
    for sel in selectors:
        elems = driver.find_elements(By.CSS_SELECTOR, sel)
        if elems:
            return elems
    try:
        pane = driver.find_element(By.CSS_SELECTOR, "#pane-side")
        driver.execute_script("arguments[0].scrollTop = 0;", pane); time.sleep(0.4)
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", pane); time.sleep(0.4)
        for sel in selectors:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            if elems:
                return elems
    except Exception:
        pass
    return []

def open_chat_by_element(driver, elem):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
        ActionChains(driver).move_to_element(elem).click(elem).perform()
        time.sleep(1.0)
    except Exception:
        try:
            elem.click(); time.sleep(1.0)
        except Exception as e:
            print("no pudo abrir chat:", e)

def get_current_chat_name(driver) -> str:
    selectors = [
        "header .copyable-text span[dir='auto']",
        "header#main header div span[title]",
        "div[data-testid='conversation-info-header'] span[dir='auto']",
        "header span[title]"
    ]
    for sel in selectors:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            name = el.get_attribute("title") or el.text
            if name:
                return sanitize_filename(name)
        except Exception:
            continue
    return f"chat_{int(time.time())}"

def get_panel_bbox(driver) -> Tuple[int,int,int,int, object]:
    selectors = ["div#main div.copyable-area", "div[role='main']", "div[role='region']"]
    for sel in selectors:
        try:
            panel = driver.find_element(By.CSS_SELECTOR, sel)
            loc = panel.location
            size = panel.size
            return int(loc['x']), int(loc['y']), int(size['width']), int(size['height']), panel
        except Exception:
            continue
    w = int(driver.execute_script("return window.innerWidth"))
    h = int(driver.execute_script("return window.innerHeight"))
    return 0, 0, w, h, None

def grid_cells_from_bbox(x, y, w, h, rows=4, cols=3):
    cells = []
    cell_w = math.ceil(w / cols)
    cell_h = math.ceil(h / rows)
    for r in range(rows):
        for c in range(cols):
            cx = x + c*cell_w
            cy = y + r*cell_h
            cells.append((cx, cy, min(cell_w, x+w-cx), min(cell_h, y+h-cy), r, c))
    return cells

def click_at(driver, abs_x, abs_y):
    try:
        body = driver.find_element(By.TAG_NAME, 'body')
        body_loc = body.location
        ActionChains(driver).move_to_element_with_offset(body, abs_x - body_loc['x'], abs_y - body_loc['y']).click().perform()
        return True
    except Exception:
        try:
            driver.execute_script("""
                var ev = new MouseEvent('click', {clientX:arguments[0], clientY:arguments[1], bubbles:true});
                var el = document.elementFromPoint(arguments[0], arguments[1]);
                if(el) el.dispatchEvent(ev);
            """, abs_x, abs_y)
            return True
        except Exception:
            return False

# Find play buttons via DOM (fast)
def find_voicenote_buttons_in_chat(driver):
    panel_selectors = ["div#main div.copyable-area", "div[role='main']", "div[role='region']", "div[tabindex='-1']"]
    panel = None
    for ps in panel_selectors:
        try:
            panel = driver.find_element(By.CSS_SELECTOR, ps)
            break
        except Exception:
            pass
    if not panel:
        panel = driver
    try:
        buttons = panel.find_elements(By.CSS_SELECTOR, "button[aria-label='Reproducir mensaje de voz']")
        if buttons:
            return buttons
    except Exception:
        pass
    try:
        spans = panel.find_elements(By.CSS_SELECTOR, "span[aria-label='Mensaje de voz']")
        possible_buttons = []
        for sp in spans:
            try:
                ancestor = sp
                for _ in range(6):
                    ancestor = ancestor.find_element(By.XPATH, "..")
                btns = ancestor.find_elements(By.CSS_SELECTOR, "button[aria-label='Reproducir mensaje de voz']")
                possible_buttons.extend(btns)
            except Exception:
                continue
        return list(dict.fromkeys(possible_buttons))
    except Exception:
        return []

def get_ancestor_message_element(el):
    current = el
    for _ in range(12):
        try:
            parent = current.find_element(By.XPATH, "..")
            classes = (parent.get_attribute("class") or "").lower()
            if any(k in classes for k in ("message-out","message-in","message","msg","message-")):
                return parent
            current = parent
        except Exception:
            break
    return el

def detect_sender_from_message_elem(msg_elem):
    classes = (msg_elem.get_attribute("class") or "").lower()
    if "message-out" in classes or ("out" in classes and "in" not in classes):
        return "me"
    try:
        span = msg_elem.find_element(By.CSS_SELECTOR, "div.copyable-text")
        attr = span.get_attribute("data-pre-plain-text") or ""
        if ":" in attr and "]" in attr:
            return "contact"
    except Exception:
        pass
    return "contact"

def dataurl_to_file(dataurl: str, filepath: Path) -> Path:
    header, b64 = dataurl.split(",", 1)
    if "audio/opus" in header or "audio/ogg" in header:
        ext = ".ogg"
    elif "audio/wav" in header:
        ext = ".wav"
    elif "audio/mpeg" in header:
        ext = ".mp3"
    else:
        ext = ".bin"
    fp = filepath.with_suffix(ext)
    with open(fp, "wb") as f:
        f.write(base64.b64decode(b64))
    return fp

# Move downloaded file into storage directory and register, avoiding duplicates by hash
def move_and_register_download(downloaded_path: Path, chat_name: str, sender_dir: Path) -> Optional[Path]:
    try:
        # calculate hash
        h = sha256_file(downloaded_path)
        if h in downloaded_index:
            existing = Path(downloaded_index[h]["stored_path"])
            print(f"Archivo ya existe por hash ({h[:8]}): {existing}. Eliminando duplicado {downloaded_path}")
            try:
                downloaded_path.unlink(missing_ok=True)
            except Exception:
                pass
            return existing
        # move to storage with organized path
        sender_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{int(time.time())}_{h[:12]}{downloaded_path.suffix}"
        stored_path = sender_dir / stored_name
        shutil.move(str(downloaded_path), str(stored_path))
        # register
        downloaded_index[h] = {
            "chat": chat_name,
            "sender": sender_dir.name,
            "stored_path": str(stored_path),
            "orig_filename": downloaded_path.name,
            "timestamp": int(time.time())
        }
        save_index()
        return stored_path
    except Exception as e:
        print("Error move_and_register_download:", e)
        return None

# Attempt download via message menu (DOM)
@retry(times=MAX_RETRIES, delay=0.6, backoff=RETRY_BACKOFF, exceptions=(Exception,))
def try_download_from_message_menu(driver, msg_elem, timeout=3) -> Optional[Path]:
    before = set(os.listdir(DOWNLOAD_DIR))
    candidates = []
    try:
        possibles = msg_elem.find_elements(By.CSS_SELECTOR, "button, div[role='button'], [data-testid]")
        for el in possibles:
            try:
                aria = (el.get_attribute("aria-label") or "").lower()
                if any(word in aria for word in ("opciones","más","menu","more","options","más opciones")):
                    candidates.append(el); continue
                dataid = (el.get_attribute("data-testid") or "").lower()
                if "menu" in dataid or "message-menu" in dataid or "msg-menu" in dataid:
                    candidates.append(el); continue
                title = (el.get_attribute("title") or "").lower()
                if any(x in title for x in ("opciones","menu","más","more")):
                    candidates.append(el); continue
            except Exception:
                continue
    except Exception:
        pass
    # dedupe
    seen = set(); uniq = []
    for c in candidates:
        s = (c.id if hasattr(c, "id") else str(c))
        if s not in seen:
            seen.add(s); uniq.append(c)
    candidates = uniq
    for cand in candidates:
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", cand)
            time.sleep(0.12)
            cand.click()
            found = None
            end = time.time() + timeout
            while time.time() < end:
                menu_items = driver.find_elements(By.XPATH, "//*[contains(text(),'Descargar') or contains(text(),'Download') or contains(text(),'Download file')]")
                if menu_items:
                    found = menu_items[0]; break
                time.sleep(0.12)
            if found:
                try:
                    found.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", found)
                end2 = time.time() + 8
                while time.time() < end2:
                    after = set(os.listdir(DOWNLOAD_DIR))
                    diff = after - before
                    if diff:
                        fname = next(iter(diff))
                        return DOWNLOAD_DIR / fname
                    time.sleep(0.3)
            else:
                try:
                    ActionChains(driver).send_keys(u'\ue00c').perform()
                except Exception:
                    pass
        except Exception:
            continue
    return None

# Visual fallback: try to find play/menu icons and click them (grid + templates)
def try_actions_in_chat_area(driver) -> bool:
    x, y, w, h, panel_el = get_panel_bbox(driver)
    cells = grid_cells_from_bbox(x, y, w, h, rows=4, cols=3)
    ordered_cells = sorted(cells, key=lambda c: (0 if c[5] == 2 else 1, c[4], c[5]))
    tpl_play = TEMPLATES_DIR / "play.png"
    tpl_menu = TEMPLATES_DIR / "menu.png"
    for (cx, cy, cw, ch, r, c) in ordered_cells:
        driver.save_screenshot(str(TMP_SCREENSHOT))
        img = Image.open(TMP_SCREENSHOT).crop((cx, cy, cx + cw, cy + ch))
        cell_shot = SCREENSHOTS_DIR / f"cell_r{r}_c{c}_{int(time.time())}.png"
        img.save(cell_shot)
        if tpl_play.exists():
            plays = find_template_positions(cell_shot, tpl_play, threshold=0.78)
            if plays:
                px, py = plays[0]
                abs_x, abs_y = cx + px, cy + py
                click_at(driver, abs_x, abs_y)
                time.sleep(CLICK_WAIT)
                return True
        if tpl_menu.exists():
            menus = find_template_positions(cell_shot, tpl_menu, threshold=0.78)
            if menus:
                mx, my = menus[0]
                abs_x, abs_y = cx + mx, cy + my
                click_at(driver, abs_x, abs_y)
                time.sleep(0.3)
                return True
    return False

# Send messages (search + type)
def open_chat_by_search(driver, contact_name: str, timeout=6) -> bool:
    search_selectors = [
        "div[title='Buscar o empezar un chat']",
        "div[role='textbox'][contenteditable='true'][data-tab]",
        "input[type='search']",
        "div[contenteditable='true'][data-tab='3']"
    ]
    el = None
    for sel in search_selectors:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            break
        except Exception:
            continue
    if not el:
        return False
    try:
        el.click()
    except Exception:
        pass
    try:
        driver.execute_script("arguments[0].innerText='';", el)
    except Exception:
        try:
            el.clear()
        except Exception:
            pass
    for ch in contact_name:
        el.send_keys(ch); time.sleep(0.02)
    end = time.time() + timeout
    while time.time() < end:
        results = driver.find_elements(By.CSS_SELECTOR, "#pane-side [role='row'], #pane-side [role='button'], [data-testid='cell-frame-container']")
        for r in results:
            try:
                if contact_name.lower() in (r.text or "").lower():
                    r.click()
                    return True
            except Exception:
                continue
        time.sleep(0.25)
    return False

def send_message_text(driver, text: str) -> bool:
    possible = ["div[contenteditable='true'][data-tab='10']", "div[contenteditable='true'][data-tab='6']", "div[title='Escribe un mensaje...']", "div[role='textbox'][contenteditable='true']"]
    el = None
    for sel in possible:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            break
        except Exception:
            pass
    if not el:
        print("No se pudo localizar cuadro de entrada de texto.")
        return False
    try:
        el.click()
    except Exception:
        pass
    for ch in text:
        el.send_keys(ch); time.sleep(0.01)
    el.send_keys("\n")
    return True

def send_lovable_messages(driver):
    for name, msg in LOVABLE_CONTACTS.items():
        print("-> enviando a", name)
        ok = open_chat_by_search(driver, name, timeout=8)
        if not ok:
            print("   no encontrado:", name)
            continue
        time.sleep(0.6)
        ok2 = send_message_text(driver, msg)
        print("   enviado?", ok2)
        time.sleep(0.6)

def read_last_messages(driver, chat_name: str, n=5) -> List[str]:
    if not open_chat_by_search(driver, chat_name, timeout=6):
        print("no se pudo abrir chat", chat_name); return []
    time.sleep(0.8)
    try:
        msgs = driver.find_elements(By.CSS_SELECTOR, "div#main div.copyable-area div.message-in, div#main div.copyable-area div.message-out")
        if not msgs:
            msgs = driver.find_elements(By.CSS_SELECTOR, "div#main div.message")
    except Exception:
        msgs = driver.find_elements(By.CSS_SELECTOR, "div.message")
    texts = []
    for m in msgs[-n:]:
        try: texts.append(m.text)
        except Exception: texts.append("")
    return texts

# ------------- Template creation helpers --------------
def create_template_from_selector(driver, selector: str, out_path: Path) -> bool:
    """Captura screenshot del primer elemento que coincida con selector y guarda en out_path."""
    try:
        el = driver.find_element(By.CSS_SELECTOR, selector)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        el.screenshot(str(out_path))
        print(f"Template guardado: {out_path}")
        return True
    except Exception as e:
        print("No se pudo crear template desde selector:", e)
        return False

def create_template_from_screenshot_crop(x:int, y:int, w:int, h:int, out_path: Path) -> bool:
    """Recorta TMP_SCREENSHOT usando coordenadas y guarda template (útil si quieres recortar manualmente)."""
    if not TMP_SCREENSHOT.exists():
        print(f"No existe {TMP_SCREENSHOT}. Haz una captura completa primero.")
        return False
    try:
        img = Image.open(TMP_SCREENSHOT)
        crop = img.crop((x, y, x + w, y + h))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path)
        print(f"Template guardado: {out_path}")
        return True
    except Exception as e:
        print("Error recortando screenshot:", e)
        return False

# ------------- Conversion helper --------------
def call_converter_for_file(file_path: Path):
    if not CALL_CONVERTER:
        return
    if not Path(CONVERTER_SCRIPT).exists():
        print("No se encontró el script de conversión:", CONVERTER_SCRIPT)
        return
    # Llamada por archivo (convert_ogg_to_wav admite archivo como --input)
    cmd = ["python", CONVERTER_SCRIPT, "--input", str(file_path), "--output", str(file_path.parent), *CONVERTER_ARGS]
    try:
        subprocess.run(cmd, check=True)
        print("Llamado convertidor OK para:", file_path)
    except Exception as e:
        print("Error al llamar al convertidor:", e)

# ------------- Main scanning logic --------------
def main():
    driver = init_driver()
    try:
        wait_for_login_and_pane(driver)
        driver.execute_script(CREATE_OBJECT_URL_HOOK)
        time.sleep(0.4)
        decode_inline_base64(driver.page_source)

        if SEND_LOVABLES:
            send_lovable_messages(driver)

        chat_elems = get_chat_elements(driver)
        if not chat_elems:
            print("No se encontraron chats.")
            return
        to_scan = chat_elems[:CHATS_TO_SCAN]
        print(f"Comenzando scan de {len(to_scan)} chats (max {CHATS_TO_SCAN})...")
        for idx, chat_el in enumerate(to_scan, start=1):
            try:
                print(f"\n[{idx}] abriendo chat...")
                open_chat_by_element(driver, chat_el)
                time.sleep(0.9)
                chat_name = get_current_chat_name(driver)
                print("Chat:", chat_name)
                btns = find_voicenote_buttons_in_chat(driver)
                if not btns:
                    print(" - no hay botones de voz visibles por DOM; intentando fallback visual...")
                    visual_ok = try_actions_in_chat_area(driver)
                    if not visual_ok:
                        print("   -> no se encontró nada por visual.")
                        continue
                    time.sleep(0.7)

                # if visual found, we still try to harvest downloads/blobs below

                for b in (btns or []):
                    try:
                        msg_ancestor = get_ancestor_message_element(b)
                        sender = detect_sender_from_message_elem(msg_ancestor)
                        chat_dir = BASE_SAVE_DIR / sanitize_filename(chat_name)
                        sender_dir = chat_dir / sender
                        sender_dir.mkdir(parents=True, exist_ok=True)

                        # 1) try menu download
                        downloaded = try_download_from_message_menu(driver, msg_ancestor)
                        if downloaded:
                            print("   -> descargado por menú:", downloaded)
                            stored = move_and_register_download(downloaded, chat_name, sender_dir)
                            if stored:
                                call_converter_for_file(stored)
                            continue

                        # 2) fallback: play and capture blob
                        prev_keys = set(driver.execute_script("return window.__wa_get_blob_keys ? window.__wa_get_blob_keys() : []"))
                        try:
                            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", b)
                        except Exception:
                            pass
                        try:
                            b.click()
                        except Exception:
                            try:
                                ActionChains(driver).move_to_element(b).click(b).perform()
                            except Exception as e:
                                print("   no se pudo clickear play:", e); continue
                        time.sleep(CLICK_WAIT)
                        new_keys = set(driver.execute_script("return window.__wa_get_blob_keys ? window.__wa_get_blob_keys() : []")) - prev_keys
                        if not new_keys:
                            time.sleep(0.6)
                            new_keys = set(driver.execute_script("return window.__wa_get_blob_keys ? window.__wa_get_blob_keys() : []")) - prev_keys
                        if not new_keys:
                            print("   -> no generó blob tras reproducir (salto).")
                            continue
                        for key in new_keys:
                            entry = driver.execute_script("return window.__wa_get_blob_entry(arguments[0]);", key)
                            if not entry:
                                continue
                            if not entry.get("dataUrl"):
                                try:
                                    b64 = driver.execute_async_script("""
                                        const url = arguments[0];
                                        const cb = arguments[arguments.length-1];
                                        fetch(url).then(r=>r.arrayBuffer()).then(buf=>{
                                          const arr=new Uint8Array(buf); let binary=''; const CHUNK=0x8000;
                                          for(let i=0;i<arr.length;i+=CHUNK){ binary+=String.fromCharCode.apply(null,arr.subarray(i,i+CHUNK)); }
                                          cb(btoa(binary));
                                        }).catch(e=>cb(null));
                                    """, key)
                                    if b64:
                                        entry["dataUrl"] = "data:audio/ogg;base64," + b64
                                except Exception as e:
                                    print("     -> fetch fallido:", e)
                            if not entry.get("dataUrl"):
                                print("     -> blob sin dataurl, salto.")
                                continue
                            # write tmp file to downloads dir so we can compute hash uniformly
                            tmpfname = DOWNLOAD_DIR / f"tmp_blob_{int(time.time())}_{key.replace('blob:','').replace('/','_')}.ogg"
                            try:
                                header, b64 = entry["dataUrl"].split(",", 1)
                                tmpfname.write_bytes(base64.b64decode(b64))
                            except Exception as e:
                                print("     -> fallo escribiendo tmp blob:", e); continue
                            stored = move_and_register_download(tmpfname, chat_name, sender_dir)
                            if stored:
                                call_converter_for_file(stored)
                            try:
                                driver.execute_script("return window.__wa_clear_blob_entry(arguments[0]);", key)
                            except Exception:
                                pass
                    except Exception as e:
                        print(" error procesando boton:", e)
                time.sleep(0.5)
            except Exception as ex:
                print("Error con chat:", ex)
        print("\nScan terminado.")
    finally:
        if CLOSE_BROWSER_AT_END:
            try:
                driver.quit()
            except Exception:
                pass

if __name__ == "__main__":
    main()
