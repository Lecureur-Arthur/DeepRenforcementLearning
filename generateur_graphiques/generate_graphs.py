import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. FONCTIONS DE NETTOYAGE (AVEC INCERTITUDE) ---
def clean_val_err(val, is_percent=False):
    """Extrait la moyenne et l'écart-type (±) d'une cellule."""
    if pd.isna(val) or val == '' or str(val).strip() == '':
        return 0.0, 0.0
    val = str(val)
    if is_percent:
        val = val.replace('%', '')
    try:
        parts = val.split('±')
        mean = float(parts[0].strip())
        std = float(parts[1].strip()) if len(parts) > 1 else 0.0
        return mean, std
    except Exception:
        return 0.0, 0.0

def get_data(df, row, col, is_percent=False):
    return clean_val_err(df.iloc[row, col], is_percent)

# --- 2. CRÉATION DES DOSSIERS ---
BASE_DIR = "generateur_graphiques"
dossiers = ['niveau 1', 'niveau 2', 'niveau 3', 'global', 'recompenses']

# Création du dossier parent et des sous-dossiers
for d in dossiers:
    chemin_dossier = os.path.join(BASE_DIR, d)
    os.makedirs(chemin_dossier, exist_ok=True)
print(f"Dossiers créés avec succès dans le répertoire '{BASE_DIR}'.")

# --- 3. EXTRACTION DES DONNÉES DEPUIS LE FICHIER EXCEL UNIQUE ---
print("Lecture des onglets du fichier Mesure.xlsx...")
chemin_excel = "Mesure.xlsx" 

try:
    df1 = pd.read_excel(chemin_excel, sheet_name=0, header=None)
    df2 = pd.read_excel(chemin_excel, sheet_name=1, header=None)
    df3 = pd.read_excel(chemin_excel, sheet_name=2, header=None)
except Exception as e:
    print(f"❌ Erreur lors de la lecture du fichier Excel : {e}")
    print("Vérifiez que le fichier s'appelle bien 'Mesure.xlsx' et qu'il est dans le même dossier que ce script.")
    exit()

# --- NIVEAU 1 (Success Rate) ---
l1_1_lazy, e1_1_lazy = get_data(df1, 5, 4, True)
l1_1_tipsy, e1_1_tipsy = get_data(df1, 6, 4, True)
l1_1_rb, e1_1_rb = get_data(df1, 7, 4, True)
l1_1_ppo, e1_1_ppo = get_data(df1, 8, 4, True)
l1_1_dqn, e1_1_dqn = get_data(df1, 10, 4, True)

l1_3_lazy, e1_3_lazy = get_data(df1, 5, 13, True)
l1_3_tipsy, e1_3_tipsy = get_data(df1, 6, 13, True)
l1_3_rb, e1_3_rb = get_data(df1, 7, 13, True)
l1_3_ppo, e1_3_ppo = get_data(df1, 8, 13, True)
l1_3_dqn, e1_3_dqn = get_data(df1, 10, 13, True)

# --- NIVEAU 2 (Success Rate) ---
l2_1_lazy, e2_1_lazy = get_data(df2, 5, 4, True)
l2_1_tipsy, e2_1_tipsy = get_data(df2, 6, 4, True)
l2_1_rb, e2_1_rb = get_data(df2, 7, 4, True)
l2_1_ppo_dir, e2_1_ppo_dir = get_data(df2, 8, 4, True)
l2_1_ppo_cl, e2_1_ppo_cl = get_data(df2, 18, 4, True)
l2_1_ppo_scr, e2_1_ppo_scr = get_data(df2, 28, 4, True)
l2_1_dqn_dir, e2_1_dqn_dir = get_data(df2, 10, 4, True)
l2_1_dqn_cl, e2_1_dqn_cl = get_data(df2, 20, 4, True)
l2_1_dqn_scr, e2_1_dqn_scr = get_data(df2, 30, 4, True)

l2_3_lazy, e2_3_lazy = get_data(df2, 5, 13, True)
l2_3_tipsy, e2_3_tipsy = get_data(df2, 6, 13, True)
l2_3_rb, e2_3_rb = get_data(df2, 7, 13, True)
l2_3_ppo_scr, e2_3_ppo_scr = get_data(df2, 28, 13, True)
l2_3_dqn_scr, e2_3_dqn_scr = get_data(df2, 30, 13, True)

# --- NIVEAU 3 (Success Rate) ---
l3_1_lazy, e3_1_lazy = get_data(df3, 5, 4, True)
l3_1_tipsy, e3_1_tipsy = get_data(df3, 6, 4, True)
l3_1_rb, e3_1_rb = get_data(df3, 7, 4, True)
l3_1_ppo_dir, e3_1_ppo_dir = get_data(df3, 8, 4, True)
l3_1_ppo_cl, e3_1_ppo_cl = get_data(df3, 18, 4, True)
l3_1_ppo_scr, e3_1_ppo_scr = get_data(df3, 28, 4, True)
l3_1_dqn_dir, e3_1_dqn_dir = get_data(df3, 10, 4, True)
l3_1_dqn_cl, e3_1_dqn_cl = get_data(df3, 20, 4, True)
l3_1_dqn_scr, e3_1_dqn_scr = get_data(df3, 30, 4, True)

l3_3_lazy, e3_3_lazy = get_data(df3, 5, 13, True)
l3_3_tipsy, e3_3_tipsy = get_data(df3, 6, 13, True)
l3_3_rb, e3_3_rb = get_data(df3, 7, 13, True)
l3_3_ppo_scr, e3_3_ppo_scr = get_data(df3, 28, 13, True)
l3_3_dqn_scr, e3_3_dqn_scr = get_data(df3, 30, 13, True)

# --- RÉCOMPENSES (Pour le bilan global) ---
r1_1_ppo, er1_1_ppo = get_data(df1, 8, 3)
r1_1_dqn, er1_1_dqn = get_data(df1, 10, 3)
r2_1_ppo_scr, er2_1_ppo_scr = get_data(df2, 28, 3)
r2_1_dqn_scr, er2_1_dqn_scr = get_data(df2, 30, 3)
r3_1_ppo_scr, er3_1_ppo_scr = get_data(df3, 28, 3)
r3_1_dqn_scr, er3_1_dqn_scr = get_data(df3, 30, 3)

# --- 4. TRACÉ DES GRAPHIQUES ---
print("Génération des graphiques avec incertitudes...")
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

def add_labels(rects, ax):
    """Ajoute le pourcentage au-dessus de la barre."""
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

# ==========================================
# DOSSIER : NIVEAU 1
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['RuleBased', 'Lazy', 'Tipsy', 'PPO', 'DQN']
x = np.arange(len(labels))
width = 0.35

means_1 = [l1_1_rb, l1_1_lazy, l1_1_tipsy, l1_1_ppo, l1_1_dqn]
errs_1 = [e1_1_rb, e1_1_lazy, e1_1_tipsy, e1_1_ppo, e1_1_dqn]
means_3 = [l1_3_rb, l1_3_lazy, l1_3_tipsy, l1_3_ppo, l1_3_dqn]
errs_3 = [e1_3_rb, e1_3_lazy, e1_3_tipsy, e1_3_ppo, e1_3_dqn]

rects1 = ax.bar(x - width/2, means_1, width, yerr=errs_1, capsize=5, label='1 Mouton', color=colors[0], alpha=0.9)
rects2 = ax.bar(x + width/2, means_3, width, yerr=errs_3, capsize=5, label='3 Moutons', color=colors[1], alpha=0.9)

ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 1 (Statique) : 1 vs 3 Moutons', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend()
plt.ylim(0, 115)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'niveau 1', 'comparaison_1_vs_3.png'), dpi=300)
plt.close()

# ==========================================
# DOSSIER : NIVEAU 2
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
means_1 = [l2_1_rb, l2_1_lazy, l2_1_tipsy, l2_1_ppo_scr, l2_1_dqn_scr]
errs_1 = [e2_1_rb, e2_1_lazy, e2_1_tipsy, e2_1_ppo_scr, e2_1_dqn_scr]
means_3 = [l2_3_rb, l2_3_lazy, l2_3_tipsy, l2_3_ppo_scr, l2_3_dqn_scr]
errs_3 = [e2_3_rb, e2_3_lazy, e2_3_tipsy, e2_3_ppo_scr, e2_3_dqn_scr]

rects1 = ax.bar(x - width/2, means_1, width, yerr=errs_1, capsize=5, label='1 Mouton', color=colors[0], alpha=0.9)
rects2 = ax.bar(x + width/2, means_3, width, yerr=errs_3, capsize=5, label='3 Moutons', color=colors[1], alpha=0.9)
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 2 (Actif) : Apprentissage Scratch (1 vs 3 Moutons)', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend()
plt.ylim(0, 115)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'niveau 2', 'comparaison_1_vs_3_scratch.png'), dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
labels_meth = ['Direct Use\n(Zero-Shot)', 'Curriculum Learning', 'Train from Scratch']
x_meth = np.arange(len(labels_meth))
means_ppo = [l2_1_ppo_dir, l2_1_ppo_cl, l2_1_ppo_scr]
errs_ppo = [e2_1_ppo_dir, e2_1_ppo_cl, e2_1_ppo_scr]
means_dqn = [l2_1_dqn_dir, l2_1_dqn_cl, l2_1_dqn_scr]
errs_dqn = [e2_1_dqn_dir, e2_1_dqn_cl, e2_1_dqn_scr]

rects1 = ax.bar(x_meth - width/2, means_ppo, width, yerr=errs_ppo, capsize=5, label='PPO', color=colors[0], alpha=0.9)
rects2 = ax.bar(x_meth + width/2, means_dqn, width, yerr=errs_dqn, capsize=5, label='DQN', color=colors[1], alpha=0.9)
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 2 (1 Mouton) : Impact de la méthode d\'apprentissage', fontweight='bold', fontsize=14)
ax.set_xticks(x_meth)
ax.set_xticklabels(labels_meth, fontweight='bold')
ax.legend()
plt.ylim(0, 85)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'niveau 2', 'comparaison_methodes_ppo_vs_dqn.png'), dpi=300)
plt.close()

# ==========================================
# DOSSIER : NIVEAU 3
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
means_1 = [l3_1_rb, l3_1_lazy, l3_1_tipsy, l3_1_ppo_scr, l3_1_dqn_scr]
errs_1 = [e3_1_rb, e3_1_lazy, e3_1_tipsy, e3_1_ppo_scr, e3_1_dqn_scr]
means_3 = [l3_3_rb, l3_3_lazy, l3_3_tipsy, l3_3_ppo_scr, l3_3_dqn_scr]
errs_3 = [e3_3_rb, e3_3_lazy, e3_3_tipsy, e3_3_ppo_scr, e3_3_dqn_scr]

rects1 = ax.bar(x - width/2, means_1, width, yerr=errs_1, capsize=5, label='1 Mouton', color=colors[0], alpha=0.9)
rects2 = ax.bar(x + width/2, means_3, width, yerr=errs_3, capsize=5, label='3 Moutons', color=colors[1], alpha=0.9)
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 3 (Obstacle) : Apprentissage Scratch (1 vs 3 Moutons)', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend()
plt.ylim(0, 115)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'niveau 3', 'comparaison_1_vs_3_scratch.png'), dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
means_ppo = [l3_1_ppo_dir, l3_1_ppo_cl, l3_1_ppo_scr]
errs_ppo = [e3_1_ppo_dir, e3_1_ppo_cl, e3_1_ppo_scr]
means_dqn = [l3_1_dqn_dir, l3_1_dqn_cl, l3_1_dqn_scr]
errs_dqn = [e3_1_dqn_dir, e3_1_dqn_cl, e3_1_dqn_scr]

rects1 = ax.bar(x_meth - width/2, means_ppo, width, yerr=errs_ppo, capsize=5, label='PPO', color=colors[0], alpha=0.9)
rects2 = ax.bar(x_meth + width/2, means_dqn, width, yerr=errs_dqn, capsize=5, label='DQN', color=colors[1], alpha=0.9)
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 3 (1 Mouton) : Impact de la méthode d\'apprentissage', fontweight='bold', fontsize=14)
ax.set_xticks(x_meth)
ax.set_xticklabels(labels_meth, fontweight='bold')
ax.legend()
plt.ylim(0, 85)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'niveau 3', 'comparaison_methodes_ppo_vs_dqn.png'), dpi=300)
plt.close()

# ==========================================
# DOSSIER : GLOBAL (Success Rate)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
labels_glob = ['Niveau 1\n(Statique)', 'Niveau 2\n(Actif)', 'Niveau 3\n(Obstacle)']
x_glob = np.arange(len(labels_glob))

y_ppo_glob = [l1_1_ppo, l2_1_ppo_scr, l3_1_ppo_scr]
e_ppo_glob = [e1_1_ppo, e2_1_ppo_scr, e3_1_ppo_scr]
y_dqn_glob = [l1_1_dqn, l2_1_dqn_scr, l3_1_dqn_scr]
e_dqn_glob = [e1_1_dqn, e2_1_dqn_scr, e3_1_dqn_scr]

rects1 = ax.bar(x_glob - width/2, y_ppo_glob, width, yerr=e_ppo_glob, capsize=5, label='PPO (Vecteurs)', color=colors[0], alpha=0.9)
rects2 = ax.bar(x_glob + width/2, y_dqn_glob, width, yerr=e_dqn_glob, capsize=5, label='DQN (Images)', color=colors[1], alpha=0.9)

ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Bilan Global : PPO vs DQN (Apprentissage Scratch - 1 Mouton)', fontweight='bold', fontsize=14)
ax.set_xticks(x_glob)
ax.set_xticklabels(labels_glob, fontweight='bold')
ax.legend()
plt.ylim(0, 85)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'global', 'bilan_global_ppo_vs_dqn.png'), dpi=300)
plt.close()

# ==========================================
# DOSSIER : RECOMPENSES
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

y_ppo_reward = [r1_1_ppo, r2_1_ppo_scr, r3_1_ppo_scr]
err_ppo_reward = [er1_1_ppo, er2_1_ppo_scr, er3_1_ppo_scr]
y_dqn_reward = [r1_1_dqn, r2_1_dqn_scr, r3_1_dqn_scr]
err_dqn_reward = [er1_1_dqn, er2_1_dqn_scr, er3_1_dqn_scr]

rects1 = ax.bar(x_glob - width/2, y_ppo_reward, width, yerr=err_ppo_reward, capsize=5, label='PPO (Vecteurs)', color=colors[0], alpha=0.9)
rects2 = ax.bar(x_glob + width/2, y_dqn_reward, width, yerr=err_dqn_reward, capsize=5, label='DQN (Images)', color=colors[1], alpha=0.9)

ax.axhline(0, color='black', linewidth=1) # Ligne du zéro
ax.set_ylabel('Average Reward', fontweight='bold')
ax.set_title('Bilan Global : Récompenses Moyennes et Stabilité (Écart-type)', fontweight='bold', fontsize=14)
ax.set_xticks(x_glob)
ax.set_xticklabels(labels_glob, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'recompenses', 'bilan_global_recompenses_erreur.png'), dpi=300)
plt.close()

print("✅ Terminé ! Tous les graphiques ont été générés avec succès.")