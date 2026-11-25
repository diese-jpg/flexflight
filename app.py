import json
import datetime
import httpx
import streamlit as st
from google import genai
from google.genai import types
from groq import Groq

st.set_page_config(page_title="FLEXFLIGHT", page_icon="✈️", layout="centered")
st.title("✈️ FLEXFLIGHT")
st.caption("Örnek: 'Aralık ayında SAW çıkışlı Cuma akşam gidiş Pazar dönüş en ucuz 5 bilet'")

gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"]) if "GEMINI_API_KEY" in st.secrets else None
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"]) if "GROQ_API_KEY" in st.secrets else None
MCP_URL = "https://www.flightlist.io/mcp"

if "mcp_sid" not in st.session_state:
    st.session_state.mcp_sid = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def mcp_request(method, params):
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if st.session_state.mcp_sid:
        headers["Mcp-Session-Id"] = st.session_state.mcp_sid
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = httpx.post(MCP_URL, headers=headers, json=payload, timeout=20.0)
    if "mcp-session-id" in r.headers:
        st.session_state.mcp_sid = r.headers["mcp-session-id"]
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    if "text/event-stream" in r.headers.get("content-type", ""):
        for line in r.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
    return r.json()

def ensure_mcp():
    if not st.session_state.mcp_sid:
        mcp_request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "flexflight", "version": "1.0"}
        })

def call_mcp_tool(name, args, limit=5000):
    try:
        ensure_mcp()
        clean_args = {k: v for k, v in args.items() if v is not None and v != ""}
        res = mcp_request("tools/call", {"name": name, "arguments": clean_args})
        if "error" in res:
            st.session_state.mcp_sid = None
            ensure_mcp()
            res = mcp_request("tools/call", {"name": name, "arguments": clean_args})
        return json.dumps(res.get("result", res), ensure_ascii=False)[:limit]
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# Gemini için araç fonksiyonları
def search_locations(query: str, locale: str = "tr") -> str:
    """Şehir, havalimanı, ülke veya bölge ID'sini bulur."""
    return call_mcp_tool("search_locations", {"query": query, "locale": locale})

def search_flights(
    from_place: str, to_place: str = None, trip: str = "round_trip",
    depart_from: str = None, depart_to: str = None,
    return_from: str = None, return_to: str = None,
    nights_from: int = None, nights_to: int = None,
    max_stops: int = None, max_price: int = None,
    depart_time: str = None, currency: str = "TRY", locale: str = "tr"
) -> str:
    """Canlı en ucuz uçuşları arar. Anywhere için to_place boş bırak."""
    return call_mcp_tool("search_flights", {
        "from": from_place, "to": to_place, "trip": trip,
        "depart_from": depart_from, "depart_to": depart_to,
        "return_from": return_from, "return_to": return_to,
        "nights_from": nights_from, "nights_to": nights_to,
        "max_stops": max_stops, "max_price": max_price,
        "depart_time": depart_time, "currency": currency, "locale": locale
    })

def route_facts(origin: str, destination: str, locale: str = "tr") -> str:
    """İki şehir/ülke arası genel fiyat ve en ucuz ay bilgisini getirir."""
    return call_mcp_tool("route_facts", {"origin": origin, "destination": destination, "locale": locale})

# Groq için araç şeması
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_locations",
            "description": "Şehir, havalimanı, ülke veya bölge ID'sini bulur.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "locale": {"type": "string", "default": "tr"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Canlı en ucuz uçuşları arar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"}, "to": {"type": "string"},
                    "trip": {"type": "string", "enum": ["one_way", "round_trip"]},
                    "depart_from": {"type": "string"}, "depart_to": {"type": "string"},
                    "return_from": {"type": "string"}, "return_to": {"type": "string"},
                    "nights_from": {"type": "integer"}, "nights_to": {"type": "integer"},
                    "max_stops": {"type": "integer"}, "max_price": {"type": "integer"},
                    "depart_time": {"type": "string", "enum": ["morning", "afternoon", "evening", "night"]},
                    "currency": {"type": "string", "default": "TRY"}, "locale": {"type": "string", "default": "tr"}
                },
                "required": ["from"]
            }
        }
    }
]

BUGUN = datetime.date.today().isoformat()
SYSTEM_PROMPT = f"""Sen FLEXFLIGHT uçuş asistanısın. Bugünün tarihi {BUGUN}.
1. Önce mutlaka search_locations ile kalkış ve varış ID'lerini bul.
2. En fazla 5 adet search_flights çağrısı yap.
3. KESİNLİKLE TABLO KULLANMA. Sonuçları en ucuzdan pahalıya şu temiz liste formatında yaz:

1. **Fiyat: X.XXX TRY** (Kalkış → Varış)
   - **Gidiş:** Gün Ay Yıl, Saat - Saat (Direkt/Aktarmalı, Süre, Havayolu - Uçuş Kodu)
   - **Dönüş:** Gün Ay Yıl, Saat - Saat (Direkt/Aktarmalı, Süre, Havayolu - Uçuş Kodu)
   - [Rezervasyon ve Detaylar için Tıklayın](link)

4. Uzun URL'leri açıkça yazma, mutlaka tıklanabilir link yap. Fiyat veya uçuş uydurma."""

GEMINI_MODELLER = ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nereye, ne zaman uçmak istiyorsun?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Canlı uçuşlar taranıyor..."):
            yanit = None

            # 1) Önce Gemini'yi dene
            if gemini_client:
                for m in GEMINI_MODELLER:
                    try:
                        chat = gemini_client.chats.create(
                            model=m,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                tools=[search_locations, search_flights, route_facts],
                                temperature=0.2,
                            )
                        )
                        yanit = chat.send_message(prompt).text
                        break
                    except Exception:
                        continue

            # 2) Gemini yoğunsa otomatik Groq devreye girsin
            if not yanit and groq_client:
                try:
                    api_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages[-2:]
                    for _ in range(6):
                        resp = groq_client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=api_msgs,
                            tools=GROQ_TOOLS,
                            tool_choice="auto",
                            temperature=0.2,
                        )
                        msg = resp.choices[0].message
                        if msg.tool_calls:
                            api_msgs.append(msg)
                            for tc in msg.tool_calls:
                                args = json.loads(tc.function.arguments or "{}")
                                sonuc = call_mcp_tool(tc.function.name, args, limit=2500)
                                api_msgs.append({"role": "tool", "tool_call_id": tc.id, "content": sonuc})
                        else:
                            yanit = msg.content
                            break
                except Exception as e:
                    yanit = f"Bir hata oluştu: {e}"

            if not yanit:
                yanit = "Şu an tüm sunucular yoğun, birazdan tekrar dene."

            st.markdown(yanit)
            st.session_state.messages.append({"role": "assistant", "content": yanit})
