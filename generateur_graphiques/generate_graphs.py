import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. FONCTION DE NETTOYAGE ---
def clean_success(val):
    if pd.isna(val) or val == '' or str(val).strip() == '':
        return 0.0
    val = str(val)
    try:
        return float(val.replace('%', '').strip())
    except Exception:
        return 0.0

# --- 2. CRÉATION DES DOSSIERS ---
# Les dossiers seront créés là où se trouve ce script (dans 'generateur_graphiques')
dossiers = ['niveau 1', 'niveau 2', 'niveau 3', 'global']
for d in dossiers:
    os.makedirs(d, exist_ok=True)
print("Dossiers créés avec succès.")

# --- 3. EXTRACTION DES DONNÉES DEPUIS LE FICHIER EXCEL UNIQUE ---
print("Lecture du fichier Mesure.xlsx (Onglets 1, 2 et 3)...")

# On remonte d'un dossier ("../") pour trouver le fichier Excel
chemin_excel = "../Mesure.xlsx"

# sheet_name=0 lit le 1er onglet, sheet_name=1 lit le 2ème, etc.
try:
    df1 = pd.read_excel(chemin_excel, sheet_name=0, header=None)
    df2 = pd.read_excel(chemin_excel, sheet_name=1, header=None)
    df3 = pd.read_excel(chemin_excel, sheet_name=2, header=None)
except Exception as e:
    print(f"❌ Erreur lors de la lecture du fichier Excel : {e}")
    print("Vérifiez que le fichier s'appelle bien 'Mesure.xlsx' et se trouve juste au-dessus de ce dossier.")
    exit()

# Niveau 1
l1_1_lazy = clean_success(df1.iloc[5, 4])
l1_1_tipsy = clean_success(df1.iloc[6, 4])
l1_1_rb = clean_success(df1.iloc[7, 4])
l1_1_ppo = clean_success(df1.iloc[8, 4])
l1_1_dqn = clean_success(df1.iloc[10, 4])

l1_3_lazy = clean_success(df1.iloc[5, 13])
l1_3_tipsy = clean_success(df1.iloc[6, 13])
l1_3_rb = clean_success(df1.iloc[7, 13])
l1_3_ppo = clean_success(df1.iloc[8, 13])
l1_3_dqn = clean_success(df1.iloc[10, 13])

# Niveau 2
l2_1_lazy = clean_success(df2.iloc[5, 4])
l2_1_tipsy = clean_success(df2.iloc[6, 4])
l2_1_rb = clean_success(df2.iloc[7, 4])
l2_1_ppo_dir = clean_success(df2.iloc[8, 4])
l2_1_ppo_cl = clean_success(df2.iloc[18, 4])
l2_1_ppo_scr = clean_success(df2.iloc[28, 4])
l2_1_dqn_dir = clean_success(df2.iloc[10, 4])
l2_1_dqn_cl = clean_success(df2.iloc[20, 4])
l2_1_dqn_scr = clean_success(df2.iloc[30, 4])

l2_3_lazy = clean_success(df2.iloc[5, 13])
l2_3_tipsy = clean_success(df2.iloc[6, 13])
l2_3_rb = clean_success(df2.iloc[7, 13])
l2_3_ppo_scr = clean_success(df2.iloc[28, 13])
l2_3_dqn_scr = clean_success(df2.iloc[30, 13])

# Niveau 3
l3_1_lazy = clean_success(df3.iloc[5, 4])
l3_1_tipsy = clean_success(df3.iloc[6, 4])
l3_1_rb = clean_success(df3.iloc[7, 4])
l3_1_ppo_dir = clean_success(df3.iloc[8, 4])
l3_1_ppo_cl = clean_success(df3.iloc[18, 4])
l3_1_ppo_scr = clean_success(df3.iloc[28, 4])
l3_1_dqn_dir = clean_success(df3.iloc[10, 4])
l3_1_dqn_cl = clean_success(df3.iloc[20, 4])
l3_1_dqn_scr = clean_success(df3.iloc[30, 4])

l3_3_lazy = clean_success(df3.iloc[5, 13])
l3_3_tipsy = clean_success(df3.iloc[6, 13])
l3_3_rb = clean_success(df3.iloc[7, 13])
l3_3_ppo_scr = clean_success(df3.iloc[28, 13])
l3_3_dqn_scr = clean_success(df3.iloc[30, 13])

# --- 4. TRACÉ DES GRAPHIQUES ---
print("Génération des graphiques en cours...")
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

def add_labels(rects, ax):
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

# ==========================================
# DOSSIER : NIVEAU 1
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['RuleBased', 'Lazy', 'Tipsy', 'PPO', 'DQN']
x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width/2, [l1_1_rb, l1_1_lazy, l1_1_tipsy, l1_1_ppo, l1_1_dqn], width, label='1 Mouton', color=colors[0])
rects2 = ax.bar(x + width/2, [l1_3_rb, l1_3_lazy, l1_3_tipsy, l1_3_ppo, l1_3_dqn], width, label='3 Moutons', color=colors[1])

ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 1 (Statique) : 1 vs 3 Moutons', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend()
plt.ylim(0, 110)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig('niveau 1/comparaison_1_vs_3.png', dpi=300)
plt.close()

# ==========================================
# DOSSIER : NIVEAU 2
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, [l2_1_rb, l2_1_lazy, l2_1_tipsy, l2_1_ppo_scr, l2_1_dqn_scr], width, label='1 Mouton', color=colors[0])
rects2 = ax.bar(x + width/2, [l2_3_rb, l2_3_lazy, l2_3_tipsy, l2_3_ppo_scr, l2_3_dqn_scr], width, label='3 Moutons', color=colors[1])
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 2 (Actif) : Apprentissage Scratch (1 vs 3 Moutons)', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend()
plt.ylim(0, 110)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig('niveau 2/comparaison_1_vs_3_scratch.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
labels_meth = ['Direct Use (Zero-Shot)', 'Curriculum Learning', 'Train from Scratch']
x_meth = np.arange(len(labels_meth))
rects1 = ax.bar(x_meth - width/2, [l2_1_ppo_dir, l2_1_ppo_cl, l2_1_ppo_scr], width, label='PPO', color=colors[0])
rects2 = ax.bar(x_meth + width/2, [l2_1_dqn_dir, l2_1_dqn_cl, l2_1_dqn_scr], width, label='DQN', color=colors[1])
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 2 (1 Mouton) : Impact de la méthode d\'apprentissage', fontweight='bold', fontsize=14)
ax.set_xticks(x_meth)
ax.set_xticklabels(labels_meth, fontweight='bold')
ax.legend()
plt.ylim(0, 80)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig('niveau 2/comparaison_methodes_ppo_vs_dqn.png', dpi=300)
plt.close()

# ==========================================
# DOSSIER : NIVEAU 3
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, [l3_1_rb, l3_1_lazy, l3_1_tipsy, l3_1_ppo_scr, l3_1_dqn_scr], width, label='1 Mouton', color=colors[0])
rects2 = ax.bar(x + width/2, [l3_3_rb, l3_3_lazy, l3_3_tipsy, l3_3_ppo_scr, l3_3_dqn_scr], width, label='3 Moutons', color=colors[1])
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 3 (Obstacle) : Apprentissage Scratch (1 vs 3 Moutons)', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend()
plt.ylim(0, 110)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig('niveau 3/comparaison_1_vs_3_scratch.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x_meth - width/2, [l3_1_ppo_dir, l3_1_ppo_cl, l3_1_ppo_scr], width, label='PPO', color=colors[0])
rects2 = ax.bar(x_meth + width/2, [l3_1_dqn_dir, l3_1_dqn_cl, l3_1_dqn_scr], width, label='DQN', color=colors[1])
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Niveau 3 (1 Mouton) : Impact de la méthode d\'apprentissage', fontweight='bold', fontsize=14)
ax.set_xticks(x_meth)
ax.set_xticklabels(labels_meth, fontweight='bold')
ax.legend()
plt.ylim(0, 80)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig('niveau 3/comparaison_methodes_ppo_vs_dqn.png', dpi=300)
plt.close()

# ==========================================
# DOSSIER : GLOBAL
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
labels_glob = ['Niveau 1\n(Statique)', 'Niveau 2\n(Actif)', 'Niveau 3\n(Obstacle)']
x_glob = np.arange(len(labels_glob))

y_ppo_glob = [l1_1_ppo, l2_1_ppo_scr, l3_1_ppo_scr]
y_dqn_glob = [l1_1_dqn, l2_1_dqn_scr, l3_1_dqn_scr]

rects1 = ax.bar(x_glob - width/2, y_ppo_glob, width, label='PPO (Vecteurs)', color=colors[0])
rects2 = ax.bar(x_glob + width/2, y_dqn_glob, width, label='DQN (Images)', color=colors[1])

ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('Bilan Global : PPO vs DQN (Apprentissage Scratch - 1 Mouton)', fontweight='bold', fontsize=14)
ax.set_xticks(x_glob)
ax.set_xticklabels(labels_glob, fontweight='bold')
ax.legend()
plt.ylim(0, 80)
add_labels(rects1 + rects2, ax)
plt.tight_layout()
plt.savefig('global/bilan_global_ppo_vs_dqn.png', dpi=300)
plt.close()

print("✅ Terminé ! Tous les graphiques ont été générés dans leurs sous-dossiers.")