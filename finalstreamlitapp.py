
import re
import time
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from pathlib import Path

# ---------- helpers ----------
def _safe_rerun():
    """Call st.rerun() with a fallback if running on older Streamlit."""
    try:
        st.rerun()
    except AttributeError:
        # Older versions used experimental_rerun; ignore if not present
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            pass

#spotify credentials go here 
# -------------------------------------

#dataset path
CSV_PATH = r"/Users/tasi/Desktop/Capstone_Project/data/spotty.csv"
# -------------------------------------

st.set_page_config(page_title="Spotify Track Lookup + Dataset Search", layout="wide")
st.title("🎵 Spotify Track Lookup + Dataset Search")

defaults = {
    "spotify_track": None,          # dict of last fetched Spotify track JSON
    "spotify_track_id": None,       # last fetched track_id
    "ds_results": None,             # DataFrame currently shown in Section 2 (prefer when set by API)
    "ds_selected_index": None,      # selected index from results
    "q_name": "",                   # live search input: name
    "q_id": "",                     # live search input: track_id
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

_token_cache = {"access_token": None, "expires_at": 0.0}

def get_access_token() -> str:
    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    cid = CLIENT_ID.strip()
    csec = CLIENT_SECRET.strip()

    try:
        cid.encode("ascii")
        csec.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("CLIENT_ID or CLIENT_SECRET contains non-ASCII characters. Retype them manually.")

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials", "client_id": cid, "client_secret": csec},
        timeout=15,
    )
    if resp.status_code != 200:
        st.error(f"Auth failed: {resp.status_code} - {resp.text}")
        st.stop()

    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + int(data.get("expires_in", 3600)) - 60
    return _token_cache["access_token"]

def parse_track_input(s: str) -> str | None:
    s = (s or "").strip()
    m = re.match(r"^spotify:track:([A-Za-z0-9]{22})$", s)
    if m:
        return m.group(1)
    try:
        parsed = urlparse(s)
        if parsed.netloc.endswith("open.spotify.com") and parsed.path.startswith("/track/"):
            tid = parsed.path.split("/track/")[1].split("/")[0]
            if re.fullmatch(r"[A-Za-z0-9]{22}", tid):
                return tid
    except Exception:
        pass
    if re.fullmatch(r"[A-Za-z0-9]{22}", s):
        return s
    return None

def get_track(track_id: str):
    token = get_access_token()
    return requests.get(
        f"https://api.spotify.com/v1/tracks/{track_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )

@st.cache_data(show_spinner=False)
def load_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")

# Load CSV once (cached)
df = None
try:
    df = load_csv(CSV_PATH)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
###################
st.header("1) Live Spotify Lookup")

with st.form("fetch_live", clear_on_submit=False):
    user_input = st.text_input(
        "Spotify Track URL or ID",
        placeholder="e.g., https://open.spotify.com/track/254bXAqt3zP6P50BdQvEsq",
    )
    submitted_live = st.form_submit_button("Fetch from Spotify")

if submitted_live:
    track_id = parse_track_input(user_input)
    if not track_id:
        st.warning("Please enter a valid Spotify track URL/URI/ID.")
        st.stop()

    with st.spinner("Fetching data from Spotify…"):
        track_resp = get_track(track_id)

    if track_resp.status_code != 200:
        st.error(f"Track request failed ({track_resp.status_code}): {track_resp.text}")
        st.stop()

    # Cache Spotify result
    st.session_state.spotify_track = track_resp.json()
    st.session_state.spotify_track_id = track_id

    # Auto-sync the dataset search with precedence rules:
    # - If exact ID match in CSV -> force that single result below AND auto-select it
    # - Else -> clear manual filters and show full CSV
    if df is not None and "track_id" in df.columns:
        auto_res = df[df["track_id"].astype(str).str.fullmatch(track_id, case=False, na=False)]
        if not auto_res.empty:
            # show only the matched row
            st.session_state.ds_results = auto_res
            # reflect filters (so the top inputs show the source of truth)
            st.session_state.q_id = track_id
            st.session_state.q_name = ""
            # programmatically set the selectbox value and rerun to lock UI
            matched_idx = int(auto_res.index[0])
            st.session_state.ds_selected_index = matched_idx
            st.session_state["ds_selectbox_key"] = matched_idx  # set widget state directly
            _safe_rerun()
        else:
            # show all rows & clear filters and selection
            st.session_state.ds_results = df.copy()
            st.session_state.q_id = ""
            st.session_state.q_name = ""
            st.session_state.ds_selected_index = None
            # clear widget state so selectbox resets, then rerun
            if "ds_selectbox_key" in st.session_state:
                del st.session_state["ds_selectbox_key"]
            _safe_rerun()
    else:
        st.session_state.ds_results = pd.DataFrame()
        st.session_state.q_id = ""
        st.session_state.q_name = ""
        st.session_state.ds_selected_index = None
        if "ds_selectbox_key" in st.session_state:
            del st.session_state["ds_selectbox_key"]
        _safe_rerun()

if st.session_state.spotify_track:
    track = st.session_state.spotify_track

    cols = st.columns([1, 2])
    with cols[0]:
        img = (track.get("album", {}) or {}).get("images", [])
        if img:
            # Updated to new API: width="stretch"
            st.image(img[0]["url"], width="stretch")
        if track.get("preview_url"):
            st.caption("30-sec Preview")
            st.audio(track["preview_url"])

    with cols[1]:
        st.markdown(f"**Name:** {track.get('name', '—')}")
        st.markdown(f"**Artist(s):** {', '.join(a['name'] for a in track.get('artists', [])) or '—'}")
        st.markdown(f"**Album:** {(track.get('album') or {}).get('name', '—')}")
        st.markdown(f"**Release date:** {(track.get('album') or {}).get('release_date', '—')}")
        st.markdown(f"**Duration:** {round(track.get('duration_ms', 0)/1000, 1)} sec")
        st.markdown(f"**Popularity:** {track.get('popularity', '—')}")
        st.markdown(f"**Track ID:** `{track.get('id', '—')}`")

    with st.expander("Raw JSON: Track"):
        st.json(track)

st.header("2) Search for a Song in Your Dataset")

if df is not None:
    expected_cols = ["track_id", "track_name", "track_popularity", "track_duration_ms", "explicit_lyrics"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"The CSV is missing expected columns: {missing}")

    # Live-search inputs (auto updates as you type; no submit button)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.session_state.q_name = st.text_input(
            "Search by track_name (contains, case-insensitive):",
            value=st.session_state.q_name,
            key="q_name_input",
            placeholder="Type song title…",
            label_visibility="visible",
        )
    with c2:
        st.session_state.q_id = st.text_input(
            "Search by track_id (exact or contains):",
            value=st.session_state.q_id,
            key="q_id_input",
            placeholder="Paste 22-char track ID…",
            label_visibility="visible",
        )
    with c3:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if st.button("Clear search"):
            st.session_state.q_name = ""
            st.session_state.q_id = ""
            st.session_state.ds_results = None
            st.session_state.ds_selected_index = None
            if "ds_selectbox_key" in st.session_state:
                del st.session_state["ds_selectbox_key"]
            _safe_rerun()

    results = df.copy()
    if st.session_state.q_name.strip():
        results = results[results["track_name"].astype(str).str.contains(st.session_state.q_name.strip(), case=False, na=False)]
    if st.session_state.q_id.strip():
        results = results[results["track_id"].astype(str).str.contains(st.session_state.q_id.strip(), case=False, na=False)]

    prefer_api = (
        st.session_state.q_name.strip() == "" and
        isinstance(st.session_state.ds_results, pd.DataFrame) and
        (
            (st.session_state.q_id.strip() == (st.session_state.spotify_track_id or "")) or
            (st.session_state.q_id.strip() == "")
        )
    )
    results_to_show = st.session_state.ds_results if prefer_api else results

    if results_to_show.empty:
        st.info("No matching rows found in your dataset.")
    else:
        st.subheader("Search Results (from your CSV)")
        show_cols = [c for c in expected_cols if c in results_to_show.columns]
        if not show_cols:
            show_cols = list(results_to_show.columns)

        st.dataframe(results_to_show[show_cols], width="stretch")


        idx_options = list(results_to_show.index)

        if "ds_selectbox_key" in st.session_state and st.session_state["ds_selectbox_key"] not in idx_options:
            del st.session_state["ds_selectbox_key"]

        default_idx = 0
        if st.session_state.ds_selected_index in idx_options:
            default_idx = idx_options.index(st.session_state.ds_selected_index)

        selected_idx = st.selectbox(
            "Select a row to view details:",
            idx_options,
            index=default_idx,
            format_func=lambda i: f"{i} — {results_to_show.at[i, 'track_name']}" if 'track_name' in results_to_show.columns else str(i),
            key="ds_selectbox_key"
        )

       
        st.session_state.ds_selected_index = selected_idx

        st.subheader("Selected Row (from your CSV)")
        st.dataframe(results_to_show.loc[[selected_idx]], width="stretch")
else:
    st.stop()
