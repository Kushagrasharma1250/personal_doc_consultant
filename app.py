import streamlit as st
import pandas as pd
import sys
import subprocess
import importlib
import requests
from pathlib import Path

# joblib may not be installed in the running Python environment (ModuleNotFoundError).
# Attempt import and if missing show an installer in the Streamlit UI so the user
# can install it into the same Python interpreter the app is running under.
try:
    import joblib
except ModuleNotFoundError:
    joblib = None
    st.error("Required package `joblib` is not installed in this Python environment.")
    st.write("Install into this environment by running:")
    st.code(f"{sys.executable} -m pip install joblib")
    if st.button("Install joblib now"):
        with st.spinner("Installing joblib..."):
            try:
                res = subprocess.run([sys.executable, "-m", "pip", "install", "joblib"], capture_output=True, text=True)
                st.write(res.stdout or res.stderr)
                if res.returncode == 0:
                    st.success("Installed joblib — reloading module...")
                    importlib.invalidate_caches()
                    try:
                        joblib = importlib.import_module("joblib")
                    except Exception as e:
                        st.error(f"Import after install failed: {e}")
                        st.stop()
                else:
                    st.error("Failed to install joblib. See output above.")
                    st.stop()
            except Exception as e:
                st.error(f"Installation process failed: {e}")
                st.stop()

# --- Option A: Load model locally (no FastAPI) ---
# Use workspace-relative paths so app runs regardless of the current working directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
INFO_PATH = BASE_DIR / "data" / "knowledge" / "perdoc2_specific_filled.csv"

import sys
import subprocess
import importlib
import math
from typing import List, Dict


def load_model_with_help(path: str):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as mnfe:
        name = getattr(mnfe, 'name', None)
        if name == 'lightgbm' or 'lightgbm' in str(mnfe).lower():
            st.error(f"Model requires LightGBM but it's not installed in this Python: {sys.executable}")
            st.write("Install into this environment by running:")
            st.code(f"{sys.executable} -m pip install lightgbm")
            if st.button("Install lightgbm now"):
                with st.spinner("Installing lightgbm..."):
                    try:
                        res = subprocess.run([sys.executable, "-m", "pip", "install", "lightgbm"], capture_output=True, text=True)
                        st.write(res.stdout)
                        if res.returncode == 0:
                            st.success("Installed lightgbm — retrying to load model...")
                            importlib.invalidate_caches()
                            try:
                                return joblib.load(path)
                            except Exception as e:
                                st.error(f"Still failed to load model after install: {e}")
                                return None
                        else:
                            st.error(f"Failed to install lightgbm: {res.stderr}")
                            return None
                    except Exception as e:
                        st.error(f"Installation process failed: {e}")
                        return None
        # other module errors or not lightgbm
        st.error(f"Failed to load model at {path}: {mnfe}")
        return None
    except Exception as e:
        st.error(f"Failed to load model at {path}: {e}")
        return None


model = load_model_with_help(str(MODEL_PATH))
if model is None:
    st.stop()

try:
    info = pd.read_csv(str(INFO_PATH)).set_index("disease")
except Exception as e:
    st.error(f"Failed to load info CSV at {INFO_PATH}: {e}")
    info = pd.DataFrame()

# Determine expected feature names from the loaded model (best-effort)
expected_features = None
try:
    if hasattr(model, "feature_name_"):
        expected_features = list(model.feature_name_)
    elif hasattr(model, "feature_names_"):
        expected_features = list(model.feature_names_)
    elif hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "booster_"):
        try:
            expected_features = list(model.booster_.feature_name())
        except Exception:
            pass
    elif hasattr(model, "get_booster"):
        try:
            expected_features = list(model.get_booster().feature_name())
        except Exception:
            pass
except Exception:
    expected_features = None

if expected_features is None:
    # try loading a saved features file if available
    feat_file = BASE_DIR / "models" / "feature_names.txt"
    if feat_file.exists():
        expected_features = [x.strip() for x in feat_file.read_text().splitlines() if x.strip()]

if expected_features is None:
    st.warning("Could not determine model feature names. Prediction may fail if input shape doesn't match.")
else:
    st.info(f"Model expects {len(expected_features)} features")

st.title("🩺 AI Disease Consultant")

st.write("Select your symptoms and get possible disease predictions with descriptions and remedies.")

# Build symptom selection UI from model feature names when possible
symptom_features = []
if expected_features is not None:
    for f in expected_features:
        lf = f.lower()
        if "symptom" in lf or lf.startswith("rare_") or any(w in lf for w in ("fever","cough","fatigue","headache","pain","nausea","vomit","diarr")):
            symptom_features.append(f)
    # fallback: if none matched, use all features
    if not symptom_features:
        symptom_features = expected_features

if symptom_features:
    # map feature -> display name
    display_map = { (f.replace("symptom_", "").replace("rare_symptom_", "").replace("_"," ").title() if f else f): f for f in symptom_features }
    options = sorted(display_map.keys())
    selected_displays = st.multiselect("Select symptoms", options)
    # selected feature names
    selected_features = { display_map[d] for d in selected_displays }
    # build user_input dict mapping feature_name -> bool for API compatibility
    user_input = {feat: (feat in selected_features) for feat in symptom_features}
else:
    # minimal fallback
    selected_features = set()
    selected_displays = []
    user_input = {}

if st.button("Predict"):
    # Build input DataFrame matching model's expected features (if known)
    if expected_features is not None:
        X = pd.DataFrame(0, index=[0], columns=expected_features)
        matched = set()
        # If we populated `selected_features` from the UI, set those directly
        try:
            sel = selected_features
        except NameError:
            sel = set()

        for feat in sel:
            if feat in X.columns:
                X.at[0, feat] = 1
                matched.add(feat)

        if len(matched) == 0:
            st.warning("Selected symptoms didn't match any model feature names; predictions may be unreliable.")
        else:
            st.write(f"Matched features: {len(matched)}/{len(X.columns)}")
    else:
        X = pd.DataFrame([user_input])

    try:
        proba = model.predict_proba(X)[0]
        labels = model.classes_
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Top-3 predictions
    ranked = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)[:3]

    st.subheader("Top Predictions")
    for disease, p in ranked:
        st.write(f"**{disease}** — Confidence: {p:.2f}")
        if disease in info.index:
            details = info.loc[disease].to_dict()
            desc = details.get("description") or details.get("discription") or details.get("causes") or details.get("details") or "N/A"
            remedy = details.get("remedy") or details.get("treatment") or details.get("remedies") or "N/A"
            prevention = details.get("prevention") or details.get("preventions") or "N/A"
            st.write("Description:", desc)
            st.write("Remedy:", remedy)
            st.write("Prevention:", prevention)
        st.markdown("---")

# --- Option B: Call FastAPI backend instead ---
# Uncomment if you want to use FastAPI running at localhost:8000

if st.button("Predict via API"):
    payload = {"features": user_input}
    response = requests.post("http://localhost:8000/predict", json=payload)
    st.json(response.json())


# --- Reference: Nearby clinics/hospitals (Google Places via API key or Overpass fallback) ---
st.markdown("---")
st.header("Nearby Clinics & Hospitals")
st.write("Find nearby clinics and hospitals. Provide a Google Maps API key and I will estimate your location and search for nearby places.")

# Google API key input (stored in session only)
api_key = st.text_input("Google Maps API Key", type="password")
radius_km = st.slider("Search radius (km)", min_value=1, max_value=50, value=5)


def google_geolocate(key: str):
    """Use Google Geolocation API to estimate location (requires API key)."""
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={key}"
    try:
        resp = requests.post(url, json={}, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        loc = j.get("location")
        if loc:
            return float(loc.get("lat")), float(loc.get("lng"))
    except Exception as e:
        return None, str(e)
    return None, "no-location"


def google_places_nearby(key: str, lat: float, lon: float, radius_m: int):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": key,
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "keyword": "clinic|hospital|doctor|health"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


if api_key:
    if st.button("Find nearby using Google Places"):
        with st.spinner("Estimating your location via Google Geolocation..."):
            latlon, err = google_geolocate(api_key)
        if latlon is None:
            st.error(f"Geolocation failed: {err}")
        else:
            lat, lon = latlon
            with st.spinner("Searching Google Places..."):
                places_json = google_places_nearby(api_key, lat, lon, int(radius_km * 1000))
            if "error" in places_json:
                st.error(f"Places search failed: {places_json['error']}")
            else:
                items = places_json.get("results", [])
                if not items:
                    st.info("No places found by Google Places in the given radius.")
                else:
                    rows = []
                    for it in items:
                        loc = it.get("geometry", {}).get("location", {})
                        rows.append({
                            "name": it.get("name"),
                            "vicinity": it.get("vicinity"),
                            "lat": loc.get("lat"),
                            "lon": loc.get("lng"),
                            "rating": it.get("rating"),
                            "place_id": it.get("place_id")
                        })
                    df = pd.DataFrame(rows)
                    st.map(df[["lat", "lon"]])
                    df["maps_url"] = df.apply(lambda r: f"https://www.google.com/maps/search/?api=1&query=place_id:{r['place_id']}", axis=1)
                    st.write(df[["name", "vicinity", "rating", "maps_url"]].to_dict(orient="records"))
else:
    st.info("No Google API key provided — you can use the Overpass fallback below.")


# Overpass fallback (use if no API key or user prefers)
st.subheader("Overpass (OSM) fallback")
st.write("If you prefer not to use Google Places, enter coordinates or use the previously-added Overpass query.")
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude (for Overpass)", value=0.0, format="%.6f")
with col2:
    lon = st.number_input("Longitude (for Overpass)", value=0.0, format="%.6f")

@st.cache_data(ttl=3600)
def find_nearby_places_overpass(lat: float, lon: float, radius_m: int) -> List[Dict]:
    # same implementation as before but renamed to avoid conflicts
    url = "https://overpass-api.de/api/interpreter"
    q = f"[out:json];(node[~\"amenity\"~\"hospital|clinic|doctors|healthcare|pharmacy\"](around:{radius_m},{lat},{lon});way[~\"amenity\"~\"hospital|clinic|doctors|healthcare|pharmacy\"](around:{radius_m},{lat},{lon});relation[~\"amenity\"~\"hospital|clinic|doctors|healthcare|pharmacy\"](around:{radius_m},{lat},{lon}););out center;"
    try:
        resp = requests.post(url, data=q, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [{"error": str(e)}]

    results = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Unnamed")
        amenity = tags.get("amenity", tags.get("healthcare", "unknown"))
        if el.get("type") == "node":
            plat = el.get("lat")
            plon = el.get("lon")
        else:
            center = el.get("center") or {}
            plat = center.get("lat")
            plon = center.get("lon")
        try:
            R = 6371000
            phi1 = math.radians(lat)
            phi2 = math.radians(plat)
            dphi = math.radians(plat - lat)
            dlambda = math.radians(plon - lon)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        except Exception:
            dist = None

        results.append({
            "name": name,
            "amenity": amenity,
            "lat": plat,
            "lon": plon,
            "distance_m": dist,
            "address": ", ".join(v for k, v in tags.items() if k.startswith("addr:" ) )
        })

    results = [r for r in results if r.get("lat") is not None and r.get("lon") is not None]
    results.sort(key=lambda x: x.get("distance_m") or 1e9)
    return results

if st.button("Find nearby via Overpass"):
    if lat == 0.0 and lon == 0.0:
        st.warning("Please enter valid latitude and longitude for Overpass search.")
    else:
        with st.spinner("Querying OpenStreetMap..."):
            places = find_nearby_places_overpass(lat, lon, int(radius_km * 1000))
        if places and "error" in places[0]:
            st.error(f"Search failed: {places[0]['error']}")
        elif not places:
            st.info("No nearby clinics or hospitals found in the given radius.")
        else:
            df = pd.DataFrame(places)
            st.map(df[["lat", "lon"]])
            df['maps_url'] = df.apply(lambda r: f"https://www.google.com/maps/search/?api=1&query={r['lat']},{r['lon']}", axis=1)
            st.write(df[["name", "amenity", "distance_m", "address", "maps_url"]].to_dict(orient='records'))
