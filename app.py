import json
import os

import altair as alt
import boto3
import pandas as pd
import streamlit as st
from dotenv import dotenv_values, load_dotenv
from langfuse import get_client
from openai import AuthenticationError, OpenAI
from pycaret.regression import load_model, predict_model

# Wczytanie zmiennych środowiskowych z .env (Langfuse czyta LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
# LANGFUSE_HOST bezpośrednio z os.environ, więc load_dotenv() musi wykonać się przed get_client())
load_dotenv()
env = dotenv_values(".env")

# Klient Langfuse (SDK 4.x, OpenTelemetry) — auth_check() w try/except, żeby zły klucz albo
# brak sieci nie wywalał całej aplikacji od razu przy starcie (za wcześnie tu na st.warning:
# st.set_page_config() musi być pierwszym wywołaniem Streamlit, więc wynik sprawdzamy niżej).
langfuse = get_client()
try:
    _langfuse_polaczony = langfuse.auth_check()
except Exception:
    _langfuse_polaczony = False

# Konfiguracja połączenia z DigitalOcean Spaces
SPACES_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SPACES_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL_S3")

BUCKET_NAME = "maraton2024"
REMOTE_MODEL_PATH = "models/maraton2024_regression_pipeline.pkl"
LOCAL_MODEL_NAME = "maraton2024_regression_pipeline"

# Model OpenAI używany w czacie (można nadpisać zmienną OPENAI_MODEL w .env)
OPENAI_MODEL = env.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

st.set_page_config(page_title="Predyktor Czasu w Półmaratonie", page_icon="🏃‍♂️")

if not _langfuse_polaczony:
    st.warning(
        "Nie udało się połączyć z Langfuse — sprawdź LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY "
        "i LANGFUSE_HOST w pliku .env. Aplikacja działa dalej, ale bez logowania ekstrakcji danych.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# OpenAI: klucz API i klient
# ---------------------------------------------------------------------------
@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    # Klucz jest argumentem funkcji, więc cache nie miesza klientów różnych użytkowników
    return OpenAI(api_key=api_key)


# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()


# ---------------------------------------------------------------------------
# Model predykcyjny (pobierany z DigitalOcean Spaces)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model_from_spaces():
    # Pobranie pliku .pkl tylko wtedy, gdy nie ma go jeszcze na dysku
    if not os.path.exists(f"{LOCAL_MODEL_NAME}.pkl"):
        with st.spinner("Pobieranie modelu predykcyjnego z chmury..."):
            session = boto3.session.Session()
            client = session.client(
                service_name="s3",
                endpoint_url=ENDPOINT_URL,
                aws_access_key_id=SPACES_KEY,
                aws_secret_access_key=SPACES_SECRET,
            )
            client.download_file(
                Bucket=BUCKET_NAME,
                Key=REMOTE_MODEL_PATH,
                Filename=f"{LOCAL_MODEL_NAME}.pkl",
            )

    # Wczytanie modelu przez PyCaret
    return load_model(LOCAL_MODEL_NAME)


try:
    model = get_model_from_spaces()
except Exception as e:
    st.error(f"Nie udało się pobrać modelu z DigitalOcean Spaces. Błąd: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Dane biegacza: parsowanie i walidacja
# ---------------------------------------------------------------------------
DYSTANSE = [5, 10, 15, 20]

# Zbiór treningowy kończy się na mecie 21,095 km, więc model przewiduje czas PÓŁMARATONU
# (nie maratonu 42,195 km). Czasy w zbiorze są w sekundach.
DYSTANS_METY_KM = 21.0975
WIEK_MIN, WIEK_MAX = 18, 100  # w zbiorze najmłodszy uczestnik miał 18 lat
POLA = ["wiek", "plec"] + [f"czas_{d}km" for d in DYSTANSE]
ETYKIETY = {
    "wiek": "Wiek",
    "plec": "Płeć",
    "czas_5km": "Czas na 5 km",
    "czas_10km": "Czas na 10 km",
    "czas_15km": "Czas na 15 km",
    "czas_20km": "Czas na 20 km",
}

# Tempo (s/km) poza tym przedziałem uznajemy za nierealne (literówka lub pomyłka)
MIN_TEMPO_S = 150  # 2:30 min/km
MAX_TEMPO_S = 720  # 12:00 min/km


def parse_time_to_seconds(text) -> int:
    """'24:30' -> 1470, '1:41:20' -> 6080. Rzuca ValueError przy złym formacie."""
    parts = str(text).strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError("użyj formatu MM:SS lub H:MM:SS")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        raise ValueError("użyj formatu MM:SS lub H:MM:SS")
    if any(n < 0 for n in nums) or nums[-1] > 59 or (len(nums) == 3 and nums[1] > 59):
        raise ValueError("nieprawidłowa wartość minut lub sekund")
    if len(nums) == 2:
        nums = [0] + nums
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def format_seconds(total: float) -> str:
    total = int(round(total))
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def validate_times(data: dict):  # zwraca komunikat błędu (str) albo None
    """Sprawdza realność tempa i to, że czasy rosną wraz z dystansem."""
    poprzedni_d, poprzedni_t = 0, 0
    for d in DYSTANSE:
        t = data.get(f"czas_{d}km")
        if t is None:
            continue
        tempo = t / d
        if not MIN_TEMPO_S <= tempo <= MAX_TEMPO_S:
            return (
                f"Czas na {d} km ({format_seconds(t)}) daje nierealne tempo "
                f"({format_seconds(tempo)[2:]} min/km)."
            )
        odcinek = (t - poprzedni_t) / (d - poprzedni_d)
        if odcinek < MIN_TEMPO_S:
            return (
                f"Czas na {d} km ({format_seconds(t)}) musi być wyraźnie dłuższy "
                f"niż czas na {poprzedni_d} km ({format_seconds(poprzedni_t)}). "
                "Czasy mają być narastające od startu."
            )
        poprzedni_d, poprzedni_t = d, t
    return None


def brakujace(data: dict) -> list[str]:
    return [p for p in POLA if data.get(p) is None]


def apply_update(args: dict) -> dict:
    """Waliduje dane z narzędzia OpenAI i zapisuje poprawne w session_state."""
    data = st.session_state.runner_data
    kandydat = dict(data)
    bledy, zapisane = [], []

    if args.get("wiek") is not None:
        try:
            wiek = int(args["wiek"])
            if not WIEK_MIN <= wiek <= WIEK_MAX:
                raise ValueError
            kandydat["wiek"] = wiek
            zapisane.append("wiek")
        except (ValueError, TypeError):
            bledy.append(f"Wiek musi być liczbą całkowitą z zakresu {WIEK_MIN}–{WIEK_MAX}.")

    if args.get("plec") is not None:
        plec = str(args["plec"]).strip().upper()
        if plec in ("M", "K"):
            kandydat["plec"] = plec
            zapisane.append("plec")
        else:
            bledy.append("Płeć musi być 'M' (mężczyzna) lub 'K' (kobieta).")

    czasy_ok = []
    for d in DYSTANSE:
        klucz = f"czas_{d}km"
        if args.get(klucz) is None:
            continue
        try:
            kandydat[klucz] = parse_time_to_seconds(args[klucz])
            czasy_ok.append(klucz)
        except ValueError as e:
            bledy.append(f"Czas na {d} km: {e}.")

    # Czasy zapisujemy tylko wtedy, gdy całość jest spójna
    if czasy_ok:
        blad_czasow = validate_times(kandydat)
        if blad_czasow:
            bledy.append(blad_czasow)
            for k in czasy_ok:
                kandydat[k] = data.get(k)
        else:
            zapisane.extend(czasy_ok)

    st.session_state.runner_data = kandydat
    return {
        "zapisano": zapisane,
        "bledy": bledy,
        "aktualne_dane": {
            "wiek": kandydat.get("wiek"),
            "plec": kandydat.get("plec"),
            **{
                f"czas_{d}km": (
                    format_seconds(kandydat[f"czas_{d}km"])
                    if kandydat.get(f"czas_{d}km") is not None
                    else None
                )
                for d in DYSTANSE
            },
        },
        "brakuje": brakujace(kandydat),
    }


# ---------------------------------------------------------------------------
# Czat z OpenAI (function calling)
# ---------------------------------------------------------------------------
# Krok 1 (każda wiadomość): wymuszone wywołanie narzędzia, które WYCIĄGA dane z rozmowy.
EXTRACT_PROMPT = """Jesteś modułem aplikacji, który wyciąga dane biegacza z rozmowy po polsku. \
Wywołaj narzędzie `zapisz_dane_biegacza` i przekaż w nim tylko to, co użytkownik podał lub poprawił \
w swojej OSTATNIEJ wiadomości. Uwzględnij kontekst rozmowy (np. samo „25:30” jako odpowiedź na \
pytanie o czas na 5 km, albo „mam 40 lat”). Jeśli nie podał niczego nowego, wywołaj narzędzie \
z pustymi argumentami. Niczego nie zgaduj.

Zasady:
- Czasy to CZASY NARASTAJĄCE OD STARTU (czas na 10 km = czas od startu do 10. kilometra), zapisuj \
je jako MM:SS lub H:MM:SS. Jeśli użytkownik poda czasy pojedynczych odcinków lub tempo, przelicz \
je ostrożnie na czasy narastające.
- plec: 'M' dla mężczyzny, 'K' dla kobiety.
- Wszystkie zmiany z jednej wypowiedzi przekaż w jednym wywołaniu.
"""

# Krok 2: rozmowa z użytkownikiem (bez narzędzi), na podstawie wyniku zapisu danych.
SYSTEM_PROMPT = """Jesteś przyjaznym asystentem aplikacji, która przewiduje czas biegacza w półmaratonie \
(21,0975 km). Rozmawiasz po polsku, krótko i konkretnie.

Twoim zadaniem jest zebranie od użytkownika (w rozmowie) sześciu informacji:
1. wiek (w latach),
2. płeć (mężczyzna / kobieta),
3. czas na 5 km,
4. czas na 10 km,
5. czas na 15 km,
6. czas na 20 km.
Czasy to CZASY NARASTAJĄCE OD STARTU (np. na 10 km: 50:30 oznacza, że po 50 min 30 s od startu \
biegacz był na 10. kilometrze), a nie czasy pojedynczych odcinków.

Dane z wiadomości użytkownika zapisuje automatycznie aplikacja. Poniżej dostajesz wynik zapisu \
ostatniej wiadomości i aktualny stan danych. Opieraj się wyłącznie na nich:
- Potwierdź krótko, co zostało zapisane. Nie twierdź, że coś zapisano, jeśli nie ma tego \
w polu „zapisano”.
- Jeśli są „bledy”, wyjaśnij użytkownikowi problem i poproś o poprawkę.
- Pytaj tylko o to, czego jeszcze „brakuje” (możesz o kilka rzeczy naraz).
- Sam NIE szacujesz wyniku półmaratonu. Gdy „brakuje” jest puste, podsumuj dane w krótkiej liście \
i poproś użytkownika o sprawdzenie. Poinformuj, że po potwierdzeniu może kliknąć przycisk \
„Oblicz wynik półmaratonu” pod czatem, a jeśli coś się nie zgadza, niech napisze poprawkę.
- Jeśli użytkownik pyta o coś niezwiązanego z biegami lub tą aplikacją, uprzejmie wróć do tematu.

Odpowiedz WYŁĄCZNIE jako obiekt JSON zgodny z podanym schematem, bez żadnego tekstu poza nim.
"""
CHAT_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "odpowiedz_asystenta",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "odpowiedz": {
                    "type": "string",
                    "description": "Wiadomość do wyświetlenia użytkownikowi w oknie czatu (markdown).",
                },
            },
            "required": ["odpowiedz"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "zapisz_dane_biegacza",
            "description": (
                "Zapisuje lub aktualizuje dane biegacza podane przez użytkownika. "
                "Przekazuj tylko te pola, które użytkownik właśnie podał lub poprawił."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "wiek": {"type": "integer", "description": "Wiek w latach (18–100)."},
                    "plec": {
                        "type": "string",
                        "enum": ["M", "K"],
                        "description": "M = mężczyzna, K = kobieta.",
                    },
                    "czas_5km": {
                        "type": "string",
                        "description": "Czas od startu do 5. km, format MM:SS lub H:MM:SS.",
                    },
                    "czas_10km": {
                        "type": "string",
                        "description": "Czas od startu do 10. km, format MM:SS lub H:MM:SS.",
                    },
                    "czas_15km": {
                        "type": "string",
                        "description": "Czas od startu do 15. km, format MM:SS lub H:MM:SS.",
                    },
                    "czas_20km": {
                        "type": "string",
                        "description": "Czas od startu do 20. km, format MM:SS lub H:MM:SS.",
                    },
                },
                "additionalProperties": False,
            },
        },
    }
]

GREETING = (
    "Cześć! 👋 Pomogę Ci oszacować czas w półmaratonie. Powiedz mi, ile masz lat, jakiej jesteś płci "
    "oraz jakie były Twoje czasy (liczone od startu) na 5, 10, 15 i 20 km. "
    "Możesz podać wszystko naraz albo krok po kroku."
)


def stan_danych() -> dict:
    """Aktualny stan zebranych danych w czytelnej formie (dla modelu i diagnostyki)."""
    dane = st.session_state.runner_data
    return {
        "zebrane": {
            k: (format_seconds(v) if k.startswith("czas_") else v)
            for k, v in dane.items()
            if v is not None
        },
        "brakuje": brakujace(dane),
    }


def extract_and_save(client: OpenAI, history: list[dict]):
    """Krok 1: wymuszone wywołanie narzędzia. Zwraca (argumenty od modelu, wynik zapisu).

    To jedyne miejsce w aplikacji logowane do Langfuse: span (rola dawnego trace) -> span
    -> generation, przez start_as_current_observation() z Twojej wersji SDK (4.x, OpenTelemetry).
    Zagnieżdżanie działa przez kontekst — kolejne "with" wewnątrz poprzedniego stają się jego
    dziećmi automatycznie, bez potrzeby wołania metody na obiekcie rodzica. Context manager sam
    kończy obserwację (.end()) na wyjściu z bloku, więc wystarczy .update() przed jego końcem.
    """
    messages = [
        {
            "role": "system",
            "content": EXTRACT_PROMPT + "\nAktualny stan danych: " + json.dumps(stan_danych(), ensure_ascii=False),
        }
    ] + history[-12:]

    ostatnia_wiadomosc = history[-1]["content"] if history else None

    with langfuse.start_as_current_observation(
        as_type="span",
        name="wyciagniecie_danych_biegacza",
        input=messages,
        metadata={"tool_choice": "zapisz_dane_biegacza"},
    ) as trace_span:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="ekstrakcja",
            input={"wiadomosc_uzytkownika": ostatnia_wiadomosc},
        ) as span:
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="wywolanie-narzedzia",
                model=OPENAI_MODEL,
                input=messages,
            ) as generation:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice={"type": "function", "function": {"name": "zapisz_dane_biegacza"}},
                    parallel_tool_calls=False,
                    temperature=0,
                )
                usage = response.usage
                generation.update(
                    output=response.choices[0].message,
                    usage_details=(
                        {"input": usage.prompt_tokens, "output": usage.completion_tokens, "total": usage.total_tokens}
                        if usage
                        else None
                    ),
                )

            calls = response.choices[0].message.tool_calls or []
            args = {}
            if calls:
                try:
                    args = json.loads(calls[0].function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            wynik = apply_update(args)
            span.update(output=wynik)

        trace_span.update(output=wynik)

    return args, wynik


def run_chat_turn(client: OpenAI, history: list[dict]) -> str:
    """Zapisuje dane z ostatniej wiadomości, a potem generuje odpowiedź asystenta.

    To wywołanie (odpowiedź czatu) NIE trafia do Langfuse — loguje się tylko krok
    wyciągania danych w extract_and_save powyżej.
    """
    args, wynik = extract_and_save(client, history)
    st.session_state.last_debug = {"argumenty_z_modelu": args, "wynik_zapisu": wynik}

    kontekst = {
        "wynik_zapisu_ostatniej_wiadomosci": {
            "zapisano": wynik["zapisano"],
            "bledy": wynik["bledy"],
        },
        "stan_danych": stan_danych(),
    }
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\nKontekst: "
                + json.dumps(kontekst, ensure_ascii=False),
            }
        ]
        + history,
        response_format=CHAT_RESPONSE_SCHEMA,
        temperature=0.3,
    )
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        odpowiedz = f"Asystent odmówił odpowiedzi: {message.refusal}"
    else:
        try:
            odpowiedz = json.loads(message.content or "{}").get("odpowiedz", "")
        except json.JSONDecodeError:
            # Nie powinno się zdarzyć przy Structured Outputs, ale nie chcemy wywalać UI
            odpowiedz = message.content or ""

    # Zabezpieczenie: model (zwłaszcza mniejszy, jak gpt-4o-mini) czasem twierdzi, że dane
    # zostały zapisane, mimo że walidacja w apply_update je odrzuciła (np. nierealne tempo
    # między punktami pomiarowymi). Błędy z "wynik" dołączamy zawsze, niezależnie od tego, czy
    # model sam o nich wspomniał, żeby użytkownik nigdy nie zobaczył fałszywego potwierdzenia
    # bez wyjaśnienia, dlaczego dana wartość jednak nie trafiła do sidebaru.
    nieujete_bledy = [b for b in wynik["bledy"] if b not in odpowiedz]
    if nieujete_bledy:
        odpowiedz = odpowiedz.rstrip() + "\n\n⚠️ " + "\n⚠️ ".join(nieujete_bledy)

    return odpowiedz


# ---------------------------------------------------------------------------
# Predykcja
# ---------------------------------------------------------------------------
def kategoria_wiekowa(wiek: int, plec: str) -> str:
    """Kategoria wiekowa jak w zbiorze: płeć + dekada (18-29 -> 20, 30-39 -> 30, ...)."""
    return f"{plec}{min(max(wiek // 10 * 10, 20), 80)}"


# Dokładnie te cechy (w tej kolejności) miał zbiór treningowy modelu (model_df w notebooku)
CECHY_MODELU = ["Płeć", "Kategoria wiekowa"] + [
    kol for d in DYSTANSE for kol in (f"{d} km Czas", f"{d} km Tempo")
]


def build_features(data: dict) -> pd.DataFrame:
    """Buduje jeden wiersz w formacie zbioru treningowego modelu.

    "X km Czas" to czas narastający od startu w SEKUNDACH, a "X km Tempo" to tempo w min/km
    na odcinku od poprzedniego punktu pomiarowego (dla 5 km: od startu).
    """
    wiersz = {
        "Płeć": data["plec"],  # 'M' / 'K'
        "Kategoria wiekowa": kategoria_wiekowa(data["wiek"], data["plec"]),
    }
    poprzedni_d, poprzedni_t = 0, 0.0
    for d in DYSTANSE:
        t = float(data[f"czas_{d}km"])
        wiersz[f"{d} km Czas"] = t
        wiersz[f"{d} km Tempo"] = (t - poprzedni_t) / 60.0 / (d - poprzedni_d)
        poprzedni_d, poprzedni_t = d, t
    return pd.DataFrame([wiersz])[CECHY_MODELU]


def predict_race(data: dict) -> float:
    """Zwraca przewidywany czas na mecie w SEKUNDACH (tak jak kolumna "Czas" w zbiorze)."""
    predictions = predict_model(model, data=build_features(data))
    return float(predictions["prediction_label"].values[0])


def summary_text(data: dict) -> str:
    return (
        f"- Wiek: **{data['wiek']}** (kategoria {kategoria_wiekowa(data['wiek'], data['plec'])})\n"
        f"- Płeć: **{'Mężczyzna' if data['plec'] == 'M' else 'Kobieta'}**\n"
        + "\n".join(
            f"- {ETYKIETY[f'czas_{d}km']}: **{format_seconds(data[f'czas_{d}km'])}**"
            for d in DYSTANSE
        )
    )


def tempo_wykres_df(data: dict) -> pd.DataFrame:
    """Tempo (min/km) na kolejnych odcinkach: od startu (0 km) do pierwszego podanego czasu,
    a potem między kolejnymi podanymi czasami. Dystanse bez czasu są pomijane — odcinek liczy
    się wtedy do najbliższego wcześniejszego punktu, który ma czas."""
    punkty = [(0, 0.0)] + [(d, data.get(f"czas_{d}km")) for d in DYSTANSE]
    obecne = [(d, t) for d, t in punkty if d == 0 or t is not None]
    wiersze = []
    for (d0, t0), (d1, t1) in zip(obecne, obecne[1:]):
        tempo_s = (t1 - t0) / (d1 - d0)
        wiersze.append(
            {
                "Dystans (km)": d1,
                "Tempo (min/km)": tempo_s / 60.0,
                "Tempo": format_seconds(tempo_s)[2:] + " min/km",
            }
        )
    return pd.DataFrame(wiersze)


# ---------------------------------------------------------------------------
# Interfejs Streamlit
# ---------------------------------------------------------------------------
if "runner_data" not in st.session_state:
    st.session_state.runner_data = {p: None for p in POLA}
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
if "last_predicted_data" not in st.session_state:
    st.session_state.last_predicted_data = None

st.title("🏃‍♂️ Predyktor Czasu w Półmaratonie")
st.write(
    "Porozmawiaj z asystentem: zapyta Cię o wiek, płeć oraz czasy na 5, 10, 15 i 20 km, "
    "a następnie oszacuje Twój potencjalny czas na dystansie półmaratonu (21,0975 km)."
)

# Historia rozmowy
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Nowa wiadomość od użytkownika
if prompt := st.chat_input("Napisz do asystenta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Asystent pisze..."):
            try:
                client = get_openai_client(st.session_state["openai_api_key"])
                # Do API wysyłamy rozmowę bez powitania startowego
                history = st.session_state.messages[1:]
                answer = run_chat_turn(client, history)
            except AuthenticationError:
                answer = "Klucz API OpenAI jest nieprawidłowy. Sprawdź go i spróbuj ponownie."
            except Exception as e:
                answer = f"Nie udało się połączyć z OpenAI. Błąd: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Panel boczny: podgląd zebranych danych i reset rozmowy
# (renderowany po obsłudze wiadomości, żeby pokazywał aktualny stan)
with st.sidebar:
    st.header("Zebrane dane")
    for pole in POLA:
        wartosc = st.session_state.runner_data.get(pole)
        if wartosc is None:
            tekst = "—"
        elif pole == "plec":
            tekst = "Mężczyzna" if wartosc == "M" else "Kobieta"
        elif pole.startswith("czas_"):
            tekst = format_seconds(wartosc)
        else:
            tekst = str(wartosc)
        st.write(f"**{ETYKIETY[pole]}:** {tekst}")
    braki = brakujace(st.session_state.runner_data)
    if braki:
        st.caption("Brakuje: " + ", ".join(ETYKIETY[b].lower() for b in braki))
    else:
        st.success("Komplet danych. Przycisk obliczenia jest pod czatem.")
    if st.button("Zacznij od nowa"):
        for klucz in ("runner_data", "messages", "last_predicted_data", "last_debug"):
            st.session_state.pop(klucz, None)
        st.rerun()
    st.subheader("Tempo na dystansie")
    tempo_df = tempo_wykres_df(st.session_state.runner_data)
    if tempo_df.empty:
        st.caption("Podaj przynajmniej jeden czas, aby zobaczyć wykres tempa.")
    else:
        wykres_tempa = (
            alt.Chart(tempo_df)
            .mark_line(
                color="#2a78d6",
                strokeWidth=2,
                point=alt.OverlayMarkDef(color="#2a78d6", filled=True, size=80),
            )
            .encode(
                x=alt.X(
                    "Dystans (km):Q",
                    title="Dystans (km)",
                    scale=alt.Scale(domain=[0, DYSTANSE[-1]]),
                    axis=alt.Axis(values=[0] + DYSTANSE, gridColor="#e1e0d9", domainColor="#c3c2b7"),
                ),
                y=alt.Y(
                    "Tempo (min/km):Q",
                    title="Tempo (min/km)",
                    axis=alt.Axis(gridColor="#e1e0d9", domainColor="#c3c2b7"),
                ),
                tooltip=[
                    alt.Tooltip("Dystans (km):Q", title="Dystans (km)"),
                    alt.Tooltip("Tempo:N", title="Tempo"),
                ],
            )
            .properties(height=200)
        )
        st.altair_chart(wykres_tempa, use_container_width=True)

# Potwierdzenie i predykcja: dostępne, gdy komplet danych jest zebrany
dane = st.session_state.runner_data
if not brakujace(dane) and dane != st.session_state.last_predicted_data:
    with st.container(border=True):
        st.markdown("**Komplet danych. Sprawdź, czy wszystko się zgadza:**")
        st.markdown(summary_text(dane))
        st.caption("Jeśli coś jest nie tak, napisz poprawkę w czacie.")
        if st.button("Oblicz wynik półmaratonu", type="primary"):
            try:
                sekundy = predict_race(dane)
            except Exception as e:
                st.error(f"Nie udało się obliczyć predykcji. Błąd: {e}")
            else:
                st.session_state.last_predicted_data = dict(dane)
                srednie_tempo = sekundy / DYSTANS_METY_KM  # s/km
                ostrzezenie = ""
                # Wynik spoza sensownego przedziału zwykle oznacza złe jednostki lub cechy modelu
                if not 3000 <= sekundy <= 18000:
                    ostrzezenie = (
                        "\n\n⚠️ Ten wynik wygląda nierealnie. Sprawdź, czy model zwraca czas "
                        "w sekundach i czy aplikacja przekazuje mu te same cechy co przy treningu."
                    )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "### Twój prognozowany czas w półmaratonie: "
                            f"`{format_seconds(sekundy)}`\n\n"
                            f"Średnie tempo: **{format_seconds(srednie_tempo)[2:]} min/km**. "
                            "Prognoza zakłada odpowiednie przygotowanie wytrzymałościowe do "
                            "dystansu 21,0975 km. "
                            "Możesz poprawić dane w czacie i policzyć wynik jeszcze raz."
                            + ostrzezenie
                        ),
                    }
                )
                st.rerun()
