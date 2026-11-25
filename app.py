import json
import datetime
import httpx
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="FLEXFLIGHT", page_icon="✈️", layout="centered")
st.title("✈️ FLEXFLIGHT")
st.caption("Örnek: 'Ocak ayında SAW çıkışlı Cuma gidiş Pazar dönüş en ucuz 5 Avrupa bileti'")

API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)
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

def call_mcp_tool(name, args):
    try:
        ensure_mcp()
        clean_args = {k: v for k, v in args.items() if v is not None and v != ""}
        res = mcp_request("tools/call", {"name": name, "arguments": clean_args})
        if "error" in res:
            st.session_state.mcp_sid = None
            ensure_mcp()
            res = mcp_request("tools/call", {"name": name, "arguments": clean_args})
        return json.dumps(res.get("result", res), ensure_ascii=False)[:6000]
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

def search_locations(query: str, locale: str = "tr") -> str:
    """Şehir, havalimanı, ülke veya bölge ID'sini bulur."""
    return call_mcp_tool("search_locations", {"query": query, "locale": locale})

def search_flights(
    from_place: str,
    to_place: str = None,
    trip: str = "round_trip",
    depart_from: str = None,
    depart_to: str = None,
    return_from: str = None,
    return_to: str = None,
    nights_from: int = None,
    nights_to: int = None,
    max_stops: int = None,
    max_price: int = None,
    currency: str = "TRY",
    locale: str = "tr"
) -> str:
    """Canlı en ucuz uçuşları arar. Anywhere için to_place boş bırak."""
    return call_mcp_tool("search_flights", {
        "from": from_place, "to": to_place, "trip": trip,
        "depart_from": depart_from, "depart_to": depart_to,
        "return_from": return_from, "return_to": return_to,
        "nights_from": nights_from, "nights_to": nights_to,
        "max_stops": max_stops, "max_price": max_price,
        "currency": currency, "locale": locale
    })

def route_facts(origin: str, destination: str, locale: str = "tr") -> str:
    """İki şehir/ülke arası genel fiyat ve en ucuz ay bilgisini getirir."""
    return call_mcp_tool("route_facts", {"origin": origin, "destination": destination, "locale": locale})

BUGUN = datetime.date.today().isoformat()
SYSTEM_PROMPT = f"""Sen FLEXFLIGHT uçuş asistanısın. Bugünün tarihi {BUGUN}.
1. Önce mutlaka search_locations ile kalkış ve varış ID'lerini bul.
2. En fazla 5 adet search_flights çağrısı yap, sistemi boğma.
3. Sonuçları en ucuzdan pahalıya sırala; tarih, saat, havayolu, aktarma ve FlightList rezervasyon linkini mutlaka yaz. Sahte fiyat uydurma."""

MODELLER = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
]

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
            hatalar = []
            for m in MODELLER:
                try:
                    chat = client.chats.create(
                        model=m,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            tools=[search_locations, search_flights, route_facts],
                            temperature=0.2,
                        )
                    )
                    res = chat.send_message(prompt)
                    yanit = res.text
                    break
                except Exception as e:
                    hatalar.append(f"**{m}:** {str(e)[:120]}")
                    continue
            if not yanit:
                yanit = "Hiçbir model yanıt vermedi. Gerçek hata detayları:\n\n" + "\n\n".join(hatalar)
            st.markdown(yanit)
            st.session_state.messages.append({"role": "assistant", "content": yanit})
