import streamlit as st
import pandas as pd
import joblib
import requests

# --- Option A: Load model locally (no FastAPI) ---
MODEL_PATH = r"C:\Users\HP\Documents\GitHub\personal_doc_consultant\models\model.joblib"
INFO_PATH = r"C:\Users\HP\Documents\GitHub\personal_doc_consultant\data\knowledge\perdoc2_specific_filled.csv"

model = joblib.load(MODEL_PATH)
info = pd.read_csv(INFO_PATH).set_index("disease")

st.title("🩺 AI Disease Consultant")

st.write("Select your symptoms and get possible disease predictions with descriptions and remedies.")

# Example symptom inputs (replace with your dataset’s actual columns)
symptoms = ["fever", "cough", "fatigue", "chest_pain", "headache"]

user_input = {}
for symptom in symptoms:
    user_input[symptom] = st.checkbox(symptom.capitalize())

if st.button("Predict"):
    # Convert to DataFrame
    X = pd.DataFrame([user_input])
    proba = model.predict_proba(X)[0]
    labels = model.classes_

    # Top-3 predictions
    ranked = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)[:3]

    st.subheader("Top Predictions")
    for disease, p in ranked:
        st.write(f"**{disease}** — Confidence: {p:.2f}")
        if disease in info.index:
            details = info.loc[disease, ["symptom_cough","symptom_dry_cough","symptom_productive_cough","symptom_shortness_of_breath","symptom_wheezing","symptom_chest_tightness","symptom_sore_throat","symptom_runny_nose","symptom_nasal_congestion","symptom_sinus_pain","symptom_loss_of_smell","symptom_loss_of_taste","symptom_pleuritic_chest_pain","symptom_hemoptysis","symptom_exercise_dyspnea","symptom_orthopnea","symptom_chest_pain","symptom_exertional_chest_pain","symptom_palpitations","symptom_orthopnea_symptom","symptom_paroxysmal_nocturnal_dyspnea","symptom_leg_swelling","symptom_exercise_intolerance","symptom_syncope","symptom_fatigue_on_exertion","symptom_resting_tachycardia","symptom_bradycardia","symptom_clammy_skin","symptom_cyanosis","symptom_peripheral_edema","symptom_claudication","symptom_headache","symptom_migraine_symptom","symptom_dizziness","symptom_vertigo","symptom_seizures_symptom","symptom_focal_weakness","symptom_numbness","symptom_tingling","symptom_vision_blur","symptom_double_vision","symptom_speech_difficulty","symptom_confusion_symptom","symptom_memory_loss_symptom","symptom_tremor_symptom","symptom_gait_instability_symptom","symptom_ataxia_symptom","symptom_paresthesia","symptom_facial_droop","symptom_photophobia","symptom_phonophobia","symptom_muscle_weakness","symptom_spasticity","symptom_nausea","symptom_vomiting","symptom_diarrhea","symptom_constipation","symptom_abdominal_pain","symptom_abdominal_bloating","symptom_heartburn","symptom_loss_of_appetite","symptom_gi_bleeding","symptom_dark_stools","symptom_jaundice_symptom","symptom_dysphagia","symptom_early_satiety","symptom_hematemesis","symptom_melena","symptom_weight_loss_unintentional","symptom_weight_gain_symptom","symptom_bitter_regurgitation","symptom_fatigue","symptom_fever","symptom_chills","symptom_night_sweats","symptom_unintentional_weight_loss","symptom_polyuria","symptom_polydipsia","symptom_polyphagia","symptom_heat_intolerance_symptom","symptom_cold_intolerance_symptom","symptom_hair_loss_symptom","symptom_muscle_cramps_symptom","symptom_joint_pain_symptom","symptom_myalgia","symptom_arthralgia","symptom_photosensitivity_symptom","symptom_facial_puffiness","symptom_cold_extremities","symptom_hot_flashes","symptom_low_back_pain","symptom_neck_pain","symptom_joint_swelling","symptom_morning_stiffness","symptom_reduced_range_of_motion","symptom_muscle_weakness_msk","symptom_bone_pain","symptom_limp","symptom_spine_rigidity","symptom_tendon_tenderness","symptom_enthesitis","symptom_rash_maculopapular","symptom_rash_psoriasiform","symptom_urticaria","symptom_ecchymosis","symptom_petechiae","symptom_alopecia_patchy","symptom_hyperpigmentation","symptom_hypopigmentation","symptom_photosensitive_rash_derm","symptom_livedo_reticularis","symptom_digital_ulcers","symptom_skin_thickening","symptom_telangiectasia","symptom_pruritus","symptom_easy_bruising","symptom_frequent_nosebleeds","symptom_prolonged_bleeding","symptom_fatigue_anemia","symptom_shortness_of_breath_on_exertion","symptom_pallor","symptom_recurrent_infections","symptom_night_sweats_heme","symptom_lymph_node_swelling","symptom_splenomegaly_symptom","symptom_low_mood","symptom_anhedonia","symptom_insomnia","symptom_hypersomnia","symptom_anxiety_symptom","symptom_panic_attacks","symptom_obsessions","symptom_compulsions","symptom_social_withdrawal","symptom_suicidal_ideation","rare_symptom_digital_clubbing","rare_symptom_butterfly_rash","rare_symptom_malar_rash","rare_symptom_raynaud_phenomenon","rare_symptom_erythema_nodosum","rare_symptom_oral_ulcers","rare_symptom_hematuria","rare_symptom_proteinuria","rare_symptom_bloody_sputum_rare","rare_symptom_neuropathic_pain_rare","rare_symptom_foot_drop_rare","rare_symptom_optic_neuritis_rare","rare_symptom_spastic_paraparesis_rare","rare_symptom_bronchiectasis_rare","rare_symptom_steatorrhea_rare","rare_symptom_massive_splenomegaly_rare","rare_symptom_portal_hypertension_signs_rare","rare_symptom_hemolytic_anemia_rare","rare_symptom_pancytopenia_rare","rare_symptom_bone_fractures_pathologic_rare","rare_symptom_bone_marrow_failure_rare","rare_symptom_hepatosplenomegaly_rare","rare_symptom_arrhythmia_persistent_rare","rare_symptom_cardiomyopathy_rare","rare_symptom_dysautonomia_rare","rare_symptom_skin_calcifications_rare","rare_symptom_angiokeratomas_rare","rare_symptom_kyphoscoliosis_rare","rare_symptom_lens_dislocation_rare","rare_symptom_tall_stature_rare","rare_symptom_hyperextensible_skin_rare","rare_symptom_joint_hypermobility_rare","rare_symptom_macroglossia_rare","rare_symptom_hearing_loss_early_onset_rare","rare_symptom_developmental_delay_rare","rare_symptom_proximal_muscle_weakness_rare","rare_symptom_bulbar_symptoms_rare","rare_symptom_respiratory_failure_rare","rare_symptom_renal_cysts_rare"]].to_dict()
            st.write("Description:", details.get("description", "N/A"))
            st.write("Remedy:", details.get("remedy", "N/A"))
            st.write("Prevention:", details.get("prevention", "N/A"))
        st.markdown("---")

# --- Option B: Call FastAPI backend instead ---
# Uncomment if you want to use FastAPI running at localhost:8000

if st.button("Predict via API"):
    payload = {"features": user_input}
    response = requests.post("http://localhost:8000/predict", json=payload)
    st.json(response.json())
