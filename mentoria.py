# LIBRERIAS
import os
import json
import uuid
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from openai import OpenAI

# CARGA DE ENTORNO Y CLIENTE
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("Falta OPENAI_API_KEY. Crea un archivo .env con tu clave.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# CONFIG BÁSICA
st.set_page_config(page_title="MentorIA", page_icon="🧭", layout="wide")

DB_PATH = "mentoria.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS plans (
            plan_id TEXT PRIMARY KEY,
            user_id TEXT,
            goal TEXT,
            context TEXT,
            week_start TEXT,
            week_end TEXT,
            plan_json TEXT,
            created_at TEXT
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS checkins (
            checkin_id TEXT PRIMARY KEY,
            user_id TEXT,
            plan_id TEXT,
            date TEXT,
            completion INTEGER,     -- 0 a 100
            mood TEXT,              -- libre
            note TEXT,              -- reflexión
            created_at TEXT
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            user_id TEXT,
            plan_id TEXT,
            week_start TEXT,
            week_end TEXT,
            summary TEXT,
            next_week_json TEXT,
            created_at TEXT
        );
        """))
init_db()

# HELPERS
def ensure_user() -> str:
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(text("""
            INSERT OR IGNORE INTO users (user_id, name, created_at)
            VALUES (:uid, :name, :ts)
            """), {"uid": st.session_state.user_id, "name": "Anon", "ts": dt.datetime.utcnow().isoformat()})
    return st.session_state.user_id

# Llamada a OpenAI
def call_chat(messages: List[Dict[str, str]], response_format: str = "json_object") -> str:

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
        response_format={"type": response_format} if response_format else None,
    )
    return resp.choices[0].message.content

def plan_prompt(goal: str, hours_per_day: int, style: str) -> List[Dict[str, str]]:
    system = (
        "Eres MentorIA, un mentor experto en cambio de hábitos y productividad. "
        "Debes generar un plan SEMANAL con 7 días. "
        "Devuelve ESTRICTAMENTE JSON con este esquema:\n"
        "{"
        "  \"title\": str, "
        "  \"why\": str, "
        "  \"days\": ["
        "     {\"day\": 1, \"objective\": str, \"tasks\": [str,...], \"time_estimate_min\": int, \"note\": str},"
        "     ... (hasta day 7)"
        "  ],"
        "  \"motivational_quote\": str"
        "}\n"
        "Reglas: micro-hábitos, tareas accionables, lenguaje claro, tiempo total diario <= {max_time} minutos."
    ).replace("{max_time}", str(int(hours_per_day*60)))

    user = (
        f"Objetivo: {goal}\n"
        f"Tiempo disponible por día: {hours_per_day} horas.\n"
        f"Estilo preferido: {style}.\n"
        "Considera barreras comunes (procrastinación, energía variable)."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

# PROMPTS DE FEEDBACK Y ADAPTACIÓN
def feedback_prompt(entries: List[Dict[str, Any]], plan_title: str) -> List[Dict[str, str]]:
    # Normaliza a lista de dicts serializables
    try:
        serializable_entries = [
            e if isinstance(e, dict) else dict(e)
            for e in entries
        ]
    except Exception:
        # Fallback muy defensivo
        serializable_entries = [json.loads(json.dumps(e, default=str)) for e in entries]

    system = (
        "Eres MentorIA, das feedback breve, empático y accionable. "
        "Devuelve ESTRICTAMENTE JSON con este esquema:\n"
        "{"
        "  \"weekly_summary\": str,"
        "  \"reinforcements\": [str,...],"
        "  \"adjustments\": [str,...],"
        "  \"next_week_focus\": [str,...]"
        "}"
    )

    user = "Resumen de check-ins y reflexiones (últimos 7 días):\n" + json.dumps(serializable_entries, ensure_ascii=False)
    assistant = f"El plan actual es: {plan_title}."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]


def next_week_prompt(goal: str, last_feedback_json: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "Eres MentorIA. Con base en el feedback y resultados, crea un NUEVO plan semanal adaptado. "
        "Devuelve ESTRICTAMENTE JSON con el mismo esquema del plan semanal."
    )
    user = "Objetivo original: " + goal
    assistant = "Feedback y sugerencias: " + json.dumps(last_feedback_json, ensure_ascii=False)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]

def date_range_this_week() -> (str, str):
    today = dt.date.today()
    start = today - dt.timedelta(days=today.weekday())  # lunes
    end = start + dt.timedelta(days=6)
    return start.isoformat(), end.isoformat()

def safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        try:
            s2 = s.strip().strip("```json").strip("```").strip()
            return json.loads(s2)
        except Exception:
            return {"error": "JSON inválido", "raw": s}

# UI: SIDEBAR
ensure_user()
with st.sidebar:
    st.header("🧭 MentorIA")
    st.markdown("Tu mentor virtual de hábitos y productividad.")
    st.divider()
    username = st.text_input("Tu nombre (opcional):", value="Anon")
    if st.button("Guardar nombre"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE users SET name=:n WHERE user_id=:u"),
                        {"n": username, "u": st.session_state.user_id})
        st.success("Nombre guardado.")

# PESTAÑA 1: PLAN SEMANAL
st.title("🧭 MentorIA – Mentor virtual")

tab1, tab2, tab3, tab4 = st.tabs(["📅 Plan semanal", "✅ Check-in diario", "📈 Progreso", "🔄 Revisión semanal"])

with tab1:
    st.subheader("Crear / Regenerar plan")
    goal = st.text_area("¿Cuál es tu objetivo principal?", placeholder="Ej: Mejorar mi organización personal en 30 días…")
    colA, colB = st.columns(2)
    with colA:
        hours = st.slider("Tiempo disponible por día (horas)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    with colB:
        style = st.selectbox("Estilo preferido", ["Directo", "Motivacional", "Didáctico", "Minimalista"], index=1)

    if st.button("🚀 Generar plan semanal"):
        msgs = plan_prompt(goal, hours, style)
        with st.spinner("Creando plan con IA…"):
            content = call_chat(msgs, response_format="json_object")
        data = safe_json_loads(content)

        start, end = date_range_this_week()
        plan_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO plans (plan_id, user_id, goal, context, week_start, week_end, plan_json, created_at)
                VALUES (:pid, :uid, :g, :ctx, :ws, :we, :pj, :ts)
            """), {
                "pid": plan_id,
                "uid": st.session_state.user_id,
                "g": goal,
                "ctx": json.dumps({"hours": hours, "style": style}, ensure_ascii=False),
                "ws": start, "we": end,
                "pj": json.dumps(data, ensure_ascii=False),
                "ts": dt.datetime.utcnow().isoformat()
            })
        st.success("Plan generado y guardado ✅")

    # Mostrar plan más reciente
    with engine.begin() as conn:
        res = conn.execute(text("""
            SELECT plan_id, week_start, week_end, plan_json, goal
            FROM plans
            WHERE user_id=:u
            ORDER BY created_at DESC
            LIMIT 1
        """), {"u": st.session_state.user_id}).mappings().all()
    if res:
        p = res[0]
        plan = safe_json_loads(p["plan_json"])
        st.markdown(f"**Objetivo:** {p['goal']}")
        st.markdown(f"**Semana:** {p['week_start']} ➜ {p['week_end']}")
        st.markdown(f"**Título del plan:** {plan.get('title', 'Plan semanal')}")
        st.info(plan.get("why", ""))
        days = plan.get("days", [])
        df_days = pd.DataFrame(days)
        if "tasks" in df_days.columns:
            df_days["tasks"] = df_days["tasks"].apply(lambda x: "\n• " + "\n• ".join(x) if isinstance(x, list) else x)
        st.dataframe(df_days.rename(columns={
            "day":"Día", "objective":"Objetivo", "tasks":"Tareas", "time_estimate_min":"Tiempo (min)", "note":"Nota"
        }), use_container_width=True)
        st.caption(f"💬 Frase: _{plan.get('motivational_quote','')}_")
    else:
        st.info("Aún no has creado un plan. Completa el formulario y presiona “Generar plan semanal”.")

# PESTAÑA 2: CHECK-IN DIARIO
with tab2:
    st.subheader("Registra tu avance de hoy")
    today = dt.date.today().isoformat()

    # Plan actual
    with engine.begin() as conn:
        res = conn.execute(text("""
            SELECT plan_id, plan_json FROM plans
            WHERE user_id=:u
            ORDER BY created_at DESC
            LIMIT 1
        """), {"u": st.session_state.user_id}).mappings().all()
    if not res:
        st.warning("Primero crea un plan en la pestaña 'Plan semanal'.")
    else:
        p = res[0]
        plan_id = p["plan_id"]

        completion = st.slider("Porcentaje de cumplimiento hoy", 0, 100, 70, step=5)
        mood = st.selectbox("Estado de ánimo", ["😃 Excelente", "🙂 Bien", "😐 Neutral", "🙁 Bajo"], index=1)
        note = st.text_area("Reflexión breve (1–2 líneas)", placeholder="¿Qué funcionó? ¿Qué puedo mejorar mañana?")

        if st.button("💾 Guardar check-in"):
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO checkins (checkin_id, user_id, plan_id, date, completion, mood, note, created_at)
                    VALUES (:cid, :uid, :pid, :d, :c, :m, :n, :ts)
                """), {
                    "cid": str(uuid.uuid4()),
                    "uid": st.session_state.user_id,
                    "pid": plan_id,
                    "d": today,
                    "c": completion,
                    "m": mood,
                    "n": note,
                    "ts": dt.datetime.utcnow().isoformat()
                })
            st.success("Check-in guardado ✅")

        # Historial reciente
        with engine.begin() as conn:
            logs = conn.execute(text("""
                SELECT date, completion, mood, note
                FROM checkins
                WHERE user_id=:u
                ORDER BY date DESC
                LIMIT 14
            """), {"u": st.session_state.user_id}).mappings().all()
        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("Aún no tienes check-ins guardados.")

# PESTAÑA 3: PROGRESO
with tab3:
    st.subheader("Tu progreso")
    with engine.begin() as conn:
        logs = conn.execute(text("""
            SELECT date, completion, mood
            FROM checkins
            WHERE user_id=:u
            ORDER BY date ASC
        """), {"u": st.session_state.user_id}).mappings().all()

    if logs:
        df = pd.DataFrame(logs)
        df["date"] = pd.to_datetime(df["date"])
        # Gráfico de cumplimiento
        fig = px.line(df, x="date", y="completion", markers=True, title="Cumplimiento diario (%)")
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        # Distribución por estado de ánimo
        mood_counts = df["mood"].value_counts().reset_index()
        mood_counts.columns = ["mood", "count"]
        fig2 = px.bar(mood_counts, x="mood", y="count", title="Estados de ánimo (frecuencia)")
        st.plotly_chart(fig2, use_container_width=True)

        # Promedios
        avg = round(df["completion"].mean(), 1)
        st.metric("Promedio de cumplimiento", f"{avg}%")
    else:
        st.info("No hay datos de progreso aún. Registra check-ins diarios para ver métricas.")

# PESTAÑA 4: REVISIÓN SEMANAL
with tab4:
    st.subheader("Resumen y ajuste de plan (semanal)")
    # Obtener último plan y últimos 7 días de check-ins
    with engine.begin() as conn:
        res_plan = conn.execute(text("""
            SELECT plan_id, goal, plan_json, week_start, week_end
            FROM plans
            WHERE user_id=:u
            ORDER BY created_at DESC
            LIMIT 1
        """), {"u": st.session_state.user_id}).mappings().all()

    if not res_plan:
        st.warning("Primero crea un plan en la pestaña 'Plan semanal'.")
    else:
        plan_row = res_plan[0]
        plan_id = plan_row["plan_id"]
        plan_title = safe_json_loads(plan_row["plan_json"]).get("title", "Plan semanal")
        goal = plan_row["goal"]

        # Entradas últimos 7 días
        start, end = date_range_this_week()
        with engine.begin() as conn:
            entries = conn.execute(text("""
                SELECT date, completion, mood, note
                FROM checkins
                WHERE user_id=:u AND date BETWEEN :s AND :e
                ORDER BY date ASC
            """), {"u": st.session_state.user_id, "s": start, "e": end}).mappings().all()

        st.markdown(f"**Semana analizada:** {start} ➜ {end}")
        if not entries:
            st.info("No hay check-ins de esta semana. Registra algunos antes de generar el resumen.")
        else:
            df_week = pd.DataFrame(entries)
            st.dataframe(df_week, use_container_width=True)

            if st.button("🧠 Generar resumen + ajustes"):
                with st.spinner("Generando feedback con IA…"):
                    msgs = feedback_prompt(entries, plan_title)
                    content = call_chat(msgs, response_format="json_object")
                    fb = safe_json_loads(content)

                st.success("Resumen generado ✅")
                st.markdown("### 📝 Resumen semanal")
                st.markdown(fb.get("weekly_summary", ""))
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Refuerzos:**")
                    for r in fb.get("reinforcements", []):
                        st.markdown(f"- {r}")
                with col2:
                    st.markdown("**Ajustes:**")
                    for a in fb.get("adjustments", []):
                        st.markdown(f"- {a}")

                st.markdown("**Enfoque para la próxima semana:**")
                for f in fb.get("next_week_focus", []):
                    st.markdown(f"- {f}")

                # Guardar review
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO reviews (review_id, user_id, plan_id, week_start, week_end, summary, next_week_json, created_at)
                        VALUES (:rid, :uid, :pid, :ws, :we, :sum, :nwj, :ts)
                    """), {
                        "rid": str(uuid.uuid4()),
                        "uid": st.session_state.user_id,
                        "pid": plan_id,
                        "ws": start, "we": end,
                        "sum": fb.get("weekly_summary", ""),
                        "nwj": json.dumps(fb, ensure_ascii=False),
                        "ts": dt.datetime.utcnow().isoformat()
                    })

                st.divider()
                if st.button("🔄 Generar nuevo plan adaptado"):
                    with st.spinner("Creando plan adaptado con IA…"):
                        msgs2 = next_week_prompt(goal, fb)
                        content2 = call_chat(msgs2, response_format="json_object")
                        new_plan = safe_json_loads(content2)

                    next_start = (pd.to_datetime(end) + pd.Timedelta(days=1)).date().isoformat()
                    next_end = (pd.to_datetime(next_start) + pd.Timedelta(days=6)).date().isoformat()
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO plans (plan_id, user_id, goal, context, week_start, week_end, plan_json, created_at)
                            VALUES (:pid, :uid, :g, :ctx, :ws, :we, :pj, :ts)
                        """), {
                            "pid": str(uuid.uuid4()),
                            "uid": st.session_state.user_id,
                            "g": goal,
                            "ctx": json.dumps({"derived_from": plan_id}, ensure_ascii=False),
                            "ws": next_start, "we": next_end,
                            "pj": json.dumps(new_plan, ensure_ascii=False),
                            "ts": dt.datetime.utcnow().isoformat()
                        })
                    st.success("Nuevo plan generado ✅ — pásate a la pestaña ‘Plan semanal’ para verlo.")