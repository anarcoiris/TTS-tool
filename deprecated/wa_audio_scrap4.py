# wa_voicenotes_capture_plus.py
# Versión extendida: descargado vía menú "Descargar" si existe + fallback blob capture.
# También incluye funciones para enviar mensajes y leer últimos mensajes.
# Ejecuta localmente. No ejecuto esto por ti.

import base64, hashlib, json, os, re, time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import functools
import math
import cv2
import numpy as np
from PIL import Image


# -------- CONFIG ----------
BASE_SAVE_DIR = Path("wa_voicenotes")
DOWNLOAD_DIR = Path("wa_downloads")  # donde Chrome guardará descargas desde menú "Descargar"
CHATS_TO_SCAN = 20
WAIT_TIMEOUT = 60
DECODED_JS_DIR = Path("decoded_inline_js")

# Para persistir sesión y evitar QR cada vez:
CHROME_USER_DATA_DIR = r"C:\Users\soyko\AppData\Local\Google\Chrome\User Data\wa_profile"
CHROME_PROFILE = None  # ej "Profile 1"

CLOSE_BROWSER_AT_END = False
SEND_LOVABLES = True  # si quieres que envíe mensajes de cariño cuando se ejecute (usa send_lovable_messages)
# Mensajes cariñosos (puedes editarlos)
LOVABLE_CONTACTS = {
    "AA Mama": "Hola mamá ❤️ Te quiero mucho. Un abrazo grande de parte mía.",
    "Jorge": "Papá, un beso grande. Te quiero mucho.",
    "JavBitch": "¡Ey, crack! Abrazo, gracias por todo 😄",
    "mi ex mujer": "Solo quería mandarte un saludo cariñoso. 😉"
}
# ---------------------------

# prepare dirs
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
DECODED_JS_DIR.mkdir(parents=True, exist_ok=True)

downloaded_index_file = BASE_SAVE_DIR / "downloaded.json"
if downloaded_index_file.exists():
    with open(downloaded_index_file, "r", encoding="utf-8") as f:
        downloaded_index = json.load(f)
else:
    downloaded_index = {}

def find_template_positions(screenshot_path, template_path, threshold=0.75):
    img = cv2.imread(str(screenshot_path), cv2.IMREAD_UNCHANGED)
    tpl = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if img is None or tpl is None:
        return []
    # convertir a gray para robustez
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    positions = []
    h, w = tpl_gray.shape
    for pt in zip(*loc[::-1]):
        positions.append((int(pt[0] + w / 2), int(pt[1] + h / 2)))  # punto central
    # eliminar duplicados cercanos
    filtered = []
    for p in positions:
        if not any((abs(p[0]-q[0])<10 and abs(p[1]-q[1])<10) for q in filtered):
            filtered.append(p)
    return filtered

def take_panel_screenshot(driver, bbox, out_path:Path):
    x, y, w, h = bbox
    tmp = Path("tmp_whatsapp_full.png")
    driver.save_screenshot(str(tmp))
    # recortar
    img = Image.open(tmp)
    cropped = img.crop((x, y, x+w, y+h))
    cropped.save(out_path)
    return out_path

def find_play_or_menu_by_visual(driver, panel_bbox, templates_dir=Path("templates")):
    # panel_bbox = (x,y,w,h)
    # templates_dir debe contener "play.png" y "menu.png" (plantillas de iconos)
    shot = Path(f"screenshots/panel_{int(time.time())}.png")
    shot.parent.mkdir(parents=True, exist_ok=True)
    take_panel_screenshot(driver, panel_bbox, shot)
    # buscar play icon
    play_tpl = templates_dir / "play.png"
    menu_tpl = templates_dir / "menu.png"
    plays = []
    menus = []
    if play_tpl.exists():
        plays = find_template_positions(shot, play_tpl, threshold=0.77)
    if menu_tpl.exists():
        menus = find_template_positions(shot, menu_tpl, threshold=0.77)
    # transformar coordenadas locales de panel a global (abs)
    abs_plays = [(panel_bbox[0]+x, panel_bbox[1]+y) for x,y in plays]
    abs_menus = [(panel_bbox[0]+x, panel_bbox[1]+y) for x,y in menus]
    return abs_plays, abs_menus

def try_actions_in_chat_area(driver):
    x,y,w,h,panel_el = get_panel_bbox(driver)
    cells = grid_cells_from_bbox(x,y,w,h, rows=4, cols=3)
    # Prioriza columnas: suponer columna derecha (c == 2) contienen iconos/menu
    ordered_cells = sorted(cells, key=lambda c: (0 if c[5]==2 else 1, c[4], c[5]))
    for (cx, cy, cw, ch, r, c) in ordered_cells:
        # comprobar visualmente si hay play/menu en esta celda (recortar la celda y buscar)
        shot_cell = Path(f"screenshots/cell_r{r}_c{c}_{int(time.time())}.png")
        # recortamos y guardamos la celda
        driver.save_screenshot("tmp_full.png")
        img = Image.open("tmp_full.png").crop((cx, cy, cx+cw, cy+ch))
        img.save(shot_cell)
        # buscar templates en esa celda:
        plays = []
        menus = []
        tpl_dir = Path("templates")
        if (tpl_dir/"play.png").exists():
            plays = find_template_positions(shot_cell, tpl_dir/"play.png", threshold=0.78)
        if plays:
            # click en el centro del primer match relativo a la celda
            px, py = plays[0]
            abs_x, abs_y = cx + px, cy + py
            click_at(driver, abs_x, abs_y)
            time.sleep(0.6)
            return True
        # si no play, buscar menú
        if (tpl_dir/"menu.png").exists():
            menus = find_template_positions(shot_cell, tpl_dir/"menu.png", threshold=0.78)
        if menus:
            mx,my = menus[0]
            abs_x, abs_y = cx + mx, cy + my
            click_at(driver, abs_x, abs_y)
            time.sleep(0.3)
            # intentar click en texto "Descargar" con OCR o buscar el elemento DOM del menú
            return True
    return False

def retry(times=3, delay=0.5, backoff=2.0, exceptions=(Exception,)):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(1, times+1):
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

def wait_for_condition(cond_fn, timeout=10, poll=0.25):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if cond_fn():
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False

def get_panel_bbox(driver):
    # intenta varios selectores y devuelve (x, y, w, h) en coordenadas de viewport
    selectors = ["div#main div.copyable-area", "div[role='main']", "div[role='region']"]
    for sel in selectors:
        try:
            panel = driver.find_element(By.CSS_SELECTOR, sel)
            loc = panel.location
            size = panel.size
            return int(loc['x']), int(loc['y']), int(size['width']), int(size['height']), panel
        except Exception:
            continue
    # fallback: usar viewport completo
    w = driver.execute_script("return window.innerWidth")
    h = driver.execute_script("return window.innerHeight")
    return 0, 0, int(w), int(h), None

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
    # mueve el ratón y cliquea en coordenadas absolutas de la ventana (viewport)
    # Selenium no ofrece click absoluto directo, usamos move_to_element_with_offset sobre body.
    body = driver.find_element(By.TAG_NAME, 'body')
    # calcular offset relativo al elemento body
    body_loc = body.location
    # usar move_to_element_with_offset con offsets relativos al elemento:
    try:
        ActionChains(driver).move_to_element_with_offset(body, abs_x - body_loc['x'], abs_y - body_loc['y']).click().perform()
        return True
    except Exception as e:
        try:
            # fallback: ejecutar JS para crear y despachar evento click en esa posición
            driver.execute_script("""
                var ev = new MouseEvent('click', {clientX:arguments[0], clientY:arguments[1], bubbles:true});
                var el = document.elementFromPoint(arguments[0], arguments[1]);
                if(el) el.dispatchEvent(ev);
            """, abs_x, abs_y)
            return True
        except Exception:
            return False

def save_index():
    with open(downloaded_index_file, "w", encoding="utf-8") as f:
        json.dump(downloaded_index, f, indent=2, ensure_ascii=False)

def sanitize_filename(s: str) -> str:
    s = re.sub(r'[\\/*?:"<>|]', "_", s)
    s = re.sub(r'\s+', " ", s).strip()
    return s[:200]

def init_driver():
    opts = Options()
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--start-maximized")
    # prefs para descargas automáticas
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

# Hook createObjectURL (igual que antes)
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

# helpers previos (decode inline base64)
def decode_inline_base64(page_source):
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

# wait for login and pane
def wait_for_login_and_pane(driver, timeout=WAIT_TIMEOUT):
    driver.get("https://web.whatsapp.com")
    print("Abre web.whatsapp.com — escanea QR si hace falta.")
    WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#pane-side, div[role='grid'], div[role='region']")))
    start = time.time()
    while True:
        elems = driver.find_elements(By.CSS_SELECTOR, "#pane-side [role='row'], #pane-side [role='button'], #pane-side [data-testid='cell-frame-container']")
        if elems and len(elems) > 0:
            break
        if time.time() - start > timeout:
            # guardar page source para inspección
            Path("wa_page_source_debug.html").write_text(driver.page_source, encoding="utf-8")
            raise TimeoutError("No se cargó la lista de chats en el tiempo esperado. Revisar wa_page_source_debug.html")
        time.sleep(1)
    print("WhatsApp Web y lista de chats cargada.")

# chat list selectors
def get_chat_elements(driver):
    selectors = ["#pane-side div[role='row']", "#pane-side div[role='button']", "#pane-side [data-testid='cell-frame-container']", "div[role='grid'] div[role='row']"]
    for sel in selectors:
        elems = driver.find_elements(By.CSS_SELECTOR, sel)
        if elems: return elems
    # scroll retry
    try:
        pane = driver.find_element(By.CSS_SELECTOR, "#pane-side")
        driver.execute_script("arguments[0].scrollTop = 0;", pane); time.sleep(0.4)
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", pane); time.sleep(0.4)
        for sel in selectors:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            if elems: return elems
    except Exception: pass
    return []

def open_chat_by_element(driver, elem):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
        ActionChains(driver).move_to_element(elem).click(elem).perform()
        time.sleep(1.0)
    except Exception:
        try: elem.click(); time.sleep(1.0)
        except Exception as e: print("no pudo abrir chat:", e)

def get_current_chat_name(driver):
    selectors = ["header .copyable-text span[dir='auto']", "header#main header div span[title]", "div[data-testid='conversation-info-header'] span[dir='auto']", "header span[title]"]
    for sel in selectors:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            name = el.get_attribute("title") or el.text
            if name: return sanitize_filename(name)
        except Exception:
            continue
    return f"chat_{int(time.time())}"

# encontrar botones play (igual que antes)
def find_voicenote_buttons_in_chat(driver):
    panel_selectors = ["div#main div.copyable-area", "div[role='main']", "div[role='region']", "div[tabindex='-1']"]
    panel = None
    for ps in panel_selectors:
        try: panel = driver.find_element(By.CSS_SELECTOR, ps); break
        except Exception: pass
    if not panel: panel = driver
    buttons = panel.find_elements(By.CSS_SELECTOR, "button[aria-label='Reproducir mensaje de voz']")
    if buttons: return buttons
    # fallback: buscar span text "Mensaje de voz" y buscar botón cercano
    spans = panel.find_elements(By.CSS_SELECTOR, "span[aria-label='Mensaje de voz']")
    possible_buttons = []
    for sp in spans:
        try:
            ancestor = sp
            for _ in range(6): ancestor = ancestor.find_element(By.XPATH, "..")
            btns = ancestor.find_elements(By.CSS_SELECTOR, "button[aria-label='Reproducir mensaje de voz']")
            possible_buttons.extend(btns)
        except Exception: pass
    return list(dict.fromkeys(possible_buttons))

# subir al contenedor del mensaje
def get_ancestor_message_element(el):
    current = el
    for _ in range(10):
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
    if "message-out" in classes or ("out" in classes and "in" not in classes): return "me"
    # intentar data-pre-plain-text
    try:
        span = msg_elem.find_element(By.CSS_SELECTOR, "div.copyable-text")
        attr = span.get_attribute("data-pre-plain-text") or ""
        if ":" in attr and "]" in attr: return "contact"
    except Exception: pass
    return "contact"

def dataurl_to_file(dataurl: str, filepath: Path):
    header, b64 = dataurl.split(",",1)
    if "audio/opus" in header or "audio/ogg" in header: ext = ".ogg"
    elif "audio/wav" in header: ext = ".wav"
    elif "audio/mpeg" in header: ext = ".mp3"
    else: ext = ".bin"
    fp = filepath.with_suffix(ext)
    with open(fp, "wb") as f: f.write(base64.b64decode(b64))
    return fp

# ---------- NUEVA FUNCIÓN: intentar descargar desde el menú contextual del mensaje ----------
def try_download_from_message_menu(driver, msg_elem, timeout=3):
    """
    Busca un botón/elemento de 'menu' dentro del msg_elem (la flechita),
    lo clickea y busca un item cuyo texto contenga 'Descargar' o 'Download'.
    Si lo encuentra, lo clickea y espera a que el archivo aparezca en DOWNLOAD_DIR.
    Devuelve path o None.
    """
    # snapshot de files antes
    before = set(os.listdir(DOWNLOAD_DIR))
    # buscar botón de menú: heurística sobre aria-labels o data-icon
    candidates = []
    try:
        # botones en el ancestro que podrían ser el menú (flecha superior derecha)
        possibles = msg_elem.find_elements(By.CSS_SELECTOR, "button, div[role='button'], [data-testid]")
        for el in possibles:
            try:
                aria = (el.get_attribute("aria-label") or "").lower()
                if any(word in aria for word in ("opciones","más","menu","more","options")):
                    candidates.append(el); continue
                dataid = (el.get_attribute("data-testid") or "").lower()
                if "menu" in dataid or "message-menu" in dataid or "msg-menu" in dataid: candidates.append(el); continue
                # iconos con title
                try:
                    title = el.get_attribute("title") or ""
                    if any(x in title.lower() for x in ("opciones","menu","más","more")):
                        candidates.append(el); continue
                except: pass
            except Exception:
                continue
    except Exception:
        pass

    # dedupe
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        # intentar buscar un botón con svg que contenga 'tail' o 'chevron' etc
        try:
            svgs = msg_elem.find_elements(By.TAG_NAME, "svg")
            if svgs:
                # no perfecto; no hacer nada si no hay candidatos concretos
                pass
        except Exception: pass

    for cand in candidates:
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", cand)
            time.sleep(0.15)
            cand.click()
            # esperar menú: buscamos un panel que contenga un elemento con texto "Descargar" o "Download"
            found = None
            end = time.time() + timeout
            while time.time() < end:
                # buscar elementos de menu en todo el documento
                menu_items = driver.find_elements(By.XPATH, "//*[contains(text(),'Descargar') or contains(text(),'Download') or contains(text(),'Download file')]")
                if menu_items:
                    found = menu_items[0]; break
                time.sleep(0.15)
            if found:
                # click en item
                try:
                    found.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", found)
                # esperar archivo nuevo aparezca en DOWNLOAD_DIR
                end2 = time.time() + 6
                while time.time() < end2:
                    after = set(os.listdir(DOWNLOAD_DIR))
                    diff = after - before
                    if diff:
                        # tomar el primer nuevo archivo (asumimos que es la descarga)
                        fname = next(iter(diff))
                        return DOWNLOAD_DIR / fname
                    time.sleep(0.3)
            else:
                # cerrar el menú si se abrió (escape)
                try:
                    ActionChains(driver).send_keys(u'\ue00c').perform()  # ESC
                except Exception: pass
        except Exception:
            continue
    return None

# ---------- Enviar mensaje a contacto ----------
def open_chat_by_search(driver, contact_name, timeout=6):
    # intenta usar la barra de búsqueda para abrir un chat por nombre
    # probamos varios selectores para la caja de búsqueda
    search_selectors = [
        "div[title='Buscar o empezar un chat']",
        "div[role='textbox'][contenteditable='true'][data-tab]",
        "input[type='search']",
        "div[contenteditable='true'][data-tab='3']"  # puede variar
    ]
    el = None
    for sel in search_selectors:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            break
        except Exception:
            continue
    if not el:
        # fallback: pulsar la tecla para abrir buscar (Ctrl+F no funciona). Intentamos abrir directamente via URL (no fiable)
        return False
    try:
        el.click()
        el.clear()
    except Exception:
        try: driver.execute_script("arguments[0].innerText='';", el)
        except: pass
    # escribir nombre
    for ch in contact_name:
        el.send_keys(ch)
        time.sleep(0.02)
    # esperar resultados y clicar el primero
    end = time.time() + timeout
    while time.time() < end:
        results = driver.find_elements(By.CSS_SELECTOR, "#pane-side [role='row'], #pane-side [role='button'], [data-testid='cell-frame-container']")
        # intentar localizar el resultado cuyo texto contenga contact_name
        for r in results:
            try:
                if contact_name.lower() in (r.text or "").lower():
                    r.click()
                    return True
            except Exception: continue
        time.sleep(0.3)
    return False

def send_message_text(driver, text):
    # localizar cuadro de mensaje (footer) y enviar texto + Enter
    possible = ["div[contenteditable='true'][data-tab='10']", "div[contenteditable='true'][data-tab='6']", "div[title='Escribe un mensaje...']", "div[role='textbox'][contenteditable='true']"]
    el = None
    for sel in possible:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            break
        except Exception: pass
    if not el:
        print("No se pudo localizar cuadro de entrada de texto.")
        return False
    # escribe y envia
    el.click()
    for ch in text:
        el.send_keys(ch)
        time.sleep(0.01)
    # enviar (Enter)
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
        print("   enviado?" , ok2)
        time.sleep(0.6)

# ---------- leer últimos mensajes de un chat ----------
def read_last_messages(driver, chat_name, n=5):
    if not open_chat_by_search(driver, chat_name, timeout=6):
        print("no se pudo abrir chat", chat_name); return []
    time.sleep(0.8)
    # buscar elementos de mensaje dentro del panel principal
    try:
        msgs = driver.find_elements(By.CSS_SELECTOR, "div#main div.copyable-area div.message-in, div#main div.copyable-area div.message-out")
        # si no, tomar genéricos
        if not msgs:
            msgs = driver.find_elements(By.CSS_SELECTOR, "div#main div.message")
    except Exception:
        msgs = driver.find_elements(By.CSS_SELECTOR, "div.message")
    texts = []
    for m in msgs[-n:]:
        try:
            texts.append(m.text)
        except Exception: texts.append("")
    return texts

# ---------- MAIN SCAN ----------
def main():
    driver = init_driver()
    try:
        wait_for_login_and_pane(driver)
        driver.execute_script(CREATE_OBJECT_URL_HOOK)
        time.sleep(0.5)
        decode_inline_base64(driver.page_source)

        # opcional: enviar mensajes cariñosos ahora
        if SEND_LOVABLES:
            send_lovable_messages(driver)

        chat_elems = get_chat_elements(driver)
        if not chat_elems:
            print("No se encontraron chats.")
            return
        to_scan = chat_elems[:CHATS_TO_SCAN]
        for idx, chat_el in enumerate(to_scan, start=1):
            print(f"[{idx}] abriendo chat...")
            open_chat_by_element(driver, chat_el)
            time.sleep(1.0)
            chat_name = get_current_chat_name(driver)
            print("Chat:", chat_name)
            btns = find_voicenote_buttons_in_chat(driver)
            if not btns:
                print(" - no hay botones de voz visibles")
                continue
            for b in btns:
                try:
                    msg_ancestor = get_ancestor_message_element(b)
                    sender = detect_sender_from_message_elem(msg_ancestor)
                    chat_dir = BASE_SAVE_DIR / sanitize_filename(chat_name)
                    sender_dir = chat_dir / sender
                    sender_dir.mkdir(parents=True, exist_ok=True)

                    # 1) intentar descargar desde menu contextual
                    downloaded = try_download_from_message_menu(driver, msg_ancestor)
                    if downloaded:
                        print("   -> descargado por menú:", downloaded)
                        key = f"{chat_name}|menu|{downloaded.name}"
                        downloaded_index[key] = {"chat":chat_name,"sender":sender,"path":str(downloaded),"timestamp":int(time.time())}
                        save_index()
                        continue

                    # 2) fallback: reproducir (como antes) y capturar blobs
                    prev_keys = set(driver.execute_script("return window.__wa_get_blob_keys ? window.__wa_get_blob_keys() : []"))
                    try:
                        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", b)
                    except Exception: pass
                    try:
                        b.click()
                    except Exception:
                        try:
                            ActionChains(driver).move_to_element(b).click(b).perform()
                        except Exception as e:
                            print("   no se pudo clickear play:", e); continue
                    time.sleep(0.6)
                    new_keys = set(driver.execute_script("return window.__wa_get_blob_keys ? window.__wa_get_blob_keys() : []")) - prev_keys
                    if not new_keys:
                        time.sleep(0.6)
                        new_keys = set(driver.execute_script("return window.__wa_get_blob_keys ? window.__wa_get_blob_keys() : []")) - prev_keys
                    if not new_keys:
                        print("   -> no generó blob tras reproducir (salto).")
                        continue
                    for key in new_keys:
                        entry = driver.execute_script("return window.__wa_get_blob_entry(arguments[0]);", key)
                        if not entry: continue
                        if not entry.get("dataUrl"):
                            # intentar fetch
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
                        uid = hashlib.sha1((chat_name+key+str(time.time())).encode()).hexdigest()[:12]
                        basepath = sender_dir / f"{int(time.time())}_{uid}"
                        index_key = f"{chat_name}|{key}"
                        if index_key in downloaded_index:
                            print("     -> ya descargado (índice).")
                            try: driver.execute_script("return window.__wa_clear_blob_entry(arguments[0]);", key)
                            except: pass
                            continue
                        saved = dataurl_to_file(entry["dataUrl"], basepath)
                        print("     -> guardado blob:", saved)
                        downloaded_index[index_key] = {"chat":chat_name,"sender":sender,"path":str(saved),"timestamp":int(time.time())}
                        save_index()
                        try: driver.execute_script("return window.__wa_clear_blob_entry(arguments[0]);", key)
                        except: pass
                except Exception as e:
                    print(" error procesando boton:", e)
            time.sleep(0.6)
        print("Scan terminado.")
    finally:
        if CLOSE_BROWSER_AT_END:
            try: driver.quit()
            except: pass

if __name__ == "__main__":
    main()
