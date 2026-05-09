import streamlit as st
import os
import json
import base64

# --- FUNKCJA DO WYŚWIETLANIA PDF ---
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # Tworzymy osadzony obiekt PDF w HTML
    # height=800 pozwala na wygodne czytanie na większości ekranów
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Baza Programów Lekowych", page_icon="💊", layout="wide")

# --- FUNKCJE BAZOWE Z CACHOWANIEM ---
# Dekorator @st.cache_data sprawia, że JSON-y ładują się tylko raz przy starcie aplikacji.
# Dzięki temu wczytanie nawet 1000 plików to ułamek sekundy.
@st.cache_data
def wczytaj_baze(folder_json="baza_json"):
    baza = []
    lista_lekow = set() # Zbiór unikalnych leków do filtrów
    
    if not os.path.exists(folder_json):
        return baza, list(lista_lekow)
        
    for plik in os.listdir(folder_json):
        if plik.endswith(".json"):
            with open(os.path.join(folder_json, plik), 'r', encoding='utf-8') as f:
                try:
                    dane = json.load(f)
                    # Symulujemy wskaźnik wiarygodności, jeśli go jeszcze nie obliczałeś w kombajnie
                    if "wiarygodnosc" not in dane:
                        dane["wiarygodnosc"] = 95 if len(dane.get("leki", [])) > 0 else 70
                    
                    baza.append(dane)
                    
                    # Zbieramy wszystkie leki do filtrów
                    for lek in dane.get("leki", []):
                        if isinstance(lek, dict) and "nazwa" in lek:
                            lista_lekow.add(lek["nazwa"])
                except Exception:
                    pass
    
    return baza, sorted(list(lista_lekow))

# --- WCZYTANIE DANYCH ---
dane_programow, wszystkie_leki = wczytaj_baze()

if not dane_programow:
    st.warning("Brak danych! Upewnij się, że folder 'baza_json' istnieje i zawiera pliki JSON.")
    st.stop()

# --- PASEK BOCZNY (FILTROWANIE) ---
st.sidebar.title("🔍 Filtrowanie")

wybrane_leki = st.sidebar.multiselect(
    "Filtruj po substancji czynnej (Leku):",
    options=wszystkie_leki,
    placeholder="Wybierz lek..."
)

# Logika filtrowania
przefiltrowana_baza = []
for program in dane_programow:
    if wybrane_leki:
        # Sprawdzamy, czy program zawiera chociaż jeden z wybranych leków
        leki_w_programie = [lek.get("nazwa") for lek in program.get("leki", []) if isinstance(lek, dict)]
        if not any(wybrany_lek in leki_w_programie for wybrany_lek in wybrane_leki):
            continue # Pomijamy program, jeśli nie ma szukanego leku
            
    przefiltrowana_baza.append(program)

st.sidebar.markdown("---")
st.sidebar.metric("Znalezione programy", len(przefiltrowana_baza))

# --- GŁÓWNY WIDOK ---
st.title("🩺 Baza Programów Lekowych NFZ")

if not przefiltrowana_baza:
    st.info("Brak programów spełniających kryteria wyszukiwania.")
    st.stop()

# Wybór konkretnego programu z przefiltrowanej listy
nazwy_programow = [p.get("program", p.get("nazwa_pliku", "Nieznany program")) for p in przefiltrowana_baza]
wybrany_tytul = st.selectbox("Wybierz program lekowy do wyświetlenia:", nazwy_programow)

# Pobranie danych wybranego programu
aktualny_program = next((p for p in przefiltrowana_baza if p.get("program", p.get("nazwa_pliku")) == wybrany_tytul), None)

if aktualny_program:
    st.markdown("---")
    
    # 1. NAGŁÓWEK I WIARYGODNOŚĆ
    kolumna_tytul, kolumna_wiarygodnosc = st.columns([3, 1])
    with kolumna_tytul:
        st.subheader(wybrany_tytul)
    with kolumna_wiarygodnosc:
        wiarygodnosc = aktualny_program.get("wiarygodnosc", 0)
        kolor = "normal" if wiarygodnosc >= 90 else ("off" if wiarygodnosc >= 70 else "error")
        st.metric("Wiarygodność AI", f"{wiarygodnosc}%", delta="Zweryfikowane" if wiarygodnosc == 100 else "Wymaga ostrożności", delta_color=kolor)

    # 2. ZAKŁADKI (Streszczenie vs Oryginał)
    zakladka_streszczenie, zakladka_oryginal, zakladka_zgloszenia = st.tabs(["📋 Ustrukturyzowane Streszczenie", "📄 Oryginalny Dokument", "⚠️ Zgłoś Błąd"])
    
    # --- ZAKŁADKA 1: STRESZCZENIE AI ---
    with zakladka_streszczenie:
        kolumna_lewa, kolumna_prawa = st.columns(2)
        
        with kolumna_lewa:
            st.markdown("### ✅ Kryteria Kwalifikacji")
            for pkt in aktualny_program.get("kryteria_kwalifikacji", []):
                st.markdown(f"- {pkt}")
                
            st.markdown("### ❌ Kryteria Wyłączenia")
            for pkt in aktualny_program.get("kryteria_wylaczenia", []):
                st.markdown(f"- {pkt}")

        with kolumna_prawa:
            st.markdown("### 💊 Leki i Dawkowanie")
            for lek in aktualny_program.get("leki", []):
                if isinstance(lek, dict):
                    with st.expander(f"**{lek.get('nazwa', 'Nieznany lek')}**", expanded=True):
                        st.write(lek.get('dawkowanie', 'Brak danych o dawkowaniu'))
            
            st.markdown("### 🔬 Badania")
            with st.expander("Badania przy kwalifikacji"):
                for badanie in aktualny_program.get("badania_przy_kwalifikacji", []):
                    st.markdown(f"- {badanie}")
            with st.expander("Badania monitorujące"):
                for badanie in aktualny_program.get("badania_monitorujace", []):
                    st.markdown(f"- {badanie}")

    # --- ZAKŁADKA 2: ORYGINALNY DOKUMENT ---
    with zakladka_oryginal:
        nazwa_pliku = aktualny_program.get("nazwa_pliku", "")

        nazwa_pliku2 = aktualny_program.get("nazwa_pliku", "").replace(".docx", ".pdf") 
        sciezka_do_pdf = os.path.join("pliki_pdf", nazwa_pliku2)
        sciezka_do_pliku = os.path.join("..", "doks", nazwa_pliku)

        if os.path.exists(sciezka_do_pdf):
            st.subheader(f"Podgląd: {nazwa_pliku2}")
            
            # Wyświetlamy PDF
            display_pdf(sciezka_do_pdf)
            
            st.markdown("---")
            
            # Przycisk pobierania zostaje pod spodem jako backup
            with open(sciezka_do_pliku, "rb") as f:
                st.download_button(
                    label="⬇️ Pobierz plik docx na dysk",
                    data=f,
                    file_name=nazwa_pliku,
                    mime="application/pdf"
                )
        else:
            st.error(f"Nie znaleziono pliku PDF: {nazwa_pliku2}. Upewnij się, że plik znajduje się w folderze 'baza_pdf'.")
        
        # # Tworzymy pełną ścieżkę do folderu, w którym trzymasz DOCX-y
        # # Zakładając, że folder 'doks' jest poziom wyżej niż skrypt streamlit
        # sciezka_do_pliku = os.path.join("..", "doks", nazwa_pliku)

        # if nazwa_pliku and os.path.exists(sciezka_do_pliku):
        #     with open(sciezka_do_pliku, "rb") as f:
        #         st.download_button(
        #             label="⬇️ Pobierz oryginalny plik DOCX",
        #             data=f, # Przekazujemy otwarty uchwyt do pliku
        #             file_name=nazwa_pliku,
        #             mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        #         )
        # else:
        #     st.error(f"Nie znaleziono pliku: {nazwa_pliku} w lokalizacji {sciezka_do_pliku}")

    # --- ZAKŁADKA 3: ZGŁASZANIE BŁĘDÓW ---
    with zakladka_zgloszenia:
        st.markdown("### Znalazłeś błąd lub braki w streszczeniu?")
        st.write("Twój feedback pomoże nam poprawić jakość algorytmów. System źródłowy (LLM) zaktualizuje bazę na podstawie Twoich uwag.")
        
        with st.form("formularz_skargi", clear_on_submit=True):
            typ_bledu = st.selectbox("Czego dotyczy błąd?", ["Brakuje leku", "Błędne kryterium kwalifikacji", "Brakuje badania", "Inne"])
            opis_bledu = st.text_area("Szczegóły:")
            przycisk_wyslij = st.form_submit_button("Wyślij zgłoszenie", type="primary")
            
            if przycisk_wyslij:
                if opis_bledu:
                    # Tutaj logika zapisu np. do bazy SQLite lub dopisania do pliku CSV
                    st.success("Zgłoszenie zostało wysłane. Dziękujemy!")
                else:
                    st.error("Proszę wpisać szczegóły błędu.")