import json
import datetime
import httpx
import streamlit as st
from groq import Groq

st.set_page_config(page_title="FLEXFLIGHT", page_icon="✈️", layout="centered")
st.title("✈️ FLEXFLIGHT")
st.caption("Örnek: 'Ocak ayında SAW çıkışlı Cuma gidiş Pazar dönüş en ucuz 5 Avrupa bileti'")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_locations",
            "description": "Şehir, havalimanı, ülke veya bölge ID'sini bulur. Arama yapmadan önce mutlaka kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Yer adı veya kodu (Örn: Istanbul, SAW, Schengen, Kazakhstan)"},
                    "locale": {"type": "string", "default": "tr"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Canlı en ucuz uçuşları arar. Anywhere için 'to' alanını boş bırak.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "Kalkış ID (search_locations'dan gelen)"},
                    "to": {"type": "string", "description": "Varış ID (Anywhere için boş bırak)"},
                    "trip": {"type": "string", "enum": ["one_way", "round_trip"]},
                    "depart_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "depart_to": {"type": "string", "description": "YYYY-MM-DD"},
                    "return_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "return_to": {"type": "string", "description": "YYYY-MM-DD"},
                    "nights_from": {"type": "integer"},
                    "nights_to": {"type": "integer"},
                    "max_stops": {"type": "integer"},
                    "max_price": {"type": "integer"},
                    "depart_time": {"type": "string", "enum": ["morning", "afternoon", "evening", "night"]},
                    "currency": {"type": "string", "default": "TRY"},
                    "locale": {"type": "string", "default": "tr"}
                },
                "required": ["from"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_facts",
            "description": "İki şehir/ülke arası genel fiyat ve en ucuz ay bilgisini getirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "locale": {"type": "string", "default": "tr"}
                },
                "required": ["origin", "destination"]
            }
        }
    }
]

BUGUN = datetime.date.today().isoformat()
SYSTEM_PROMPT = f"""Sen FLEXFLIGHT uçuş asistanısın. Bugünün tarihi {BUGUN}.
1. Önce mutlaka search_locations ile kalkış ve varış ID'lerini bul.
2. En fazla 5 adet search_flights çağrısı yap, sistemi boğma.
3. Sonuçları en ucuzdan pahalıya sırala; tarih, saat, havayolu, aktarma ve FlightList rezervasyon linkini mutlaka paylaş. Sahte fiyat uydurma."""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nereye, ne zaman uçmak istiyorsun?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Canlı uçuşlar taranıyor..."):
            try:
                api_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages[-6:]
                yanit = "Sonuç alınamadı."
                for _ in range(8):
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=api_msgs,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=0.2,
                    )
                    msg = resp.choices[0].message
                    if msg.tool_calls:
                        api_msgs.append(msg)
                        for tc in msg.tool_calls:
                            args = json.loads(tc.function.arguments or "{}")
                            sonuc = call_mcp_tool(tc.function.name, args)
                            api_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": sonuc,
                            })
                    else:
                        yanit = msg.content
                        break
            except Exception as e:
                yanit = f"Bir hata oluştu: {e}"

            st.markdown(yanit)
            st.session_state.messages.append({"role": "assistant", "content": yanit})
