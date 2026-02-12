# A Faire :

* Average Sheep enter a pour le nombre d'episode tester (pour par exemple 100 episode combien de moouton sont entrée (A mouton par episode))

* Comparer avec les modele typsy et Lasy pour confirmer si un modele a bien apprie ou non.

* Compter le nombre de mounton entrée pour chaque épisode (exemple sur les 100 run dire qu'il a réussi a rentrée 58 moutons).

* recommencer au step 1 avec plusieurs moutons (*calculer le success rate*)



# STEP 1 : 

### python test.py -t PPO -a models/ppo_sheep1_obst0.zip -s 1 -n 100

Ran 100 episodes.
Average reward: 1174.61 ± 252.41 (95% CI)


### python test.py -t TD3 -a models/td3_sheep1_obst0.zip -s 1 -n 100

Ran 100 episodes.
Average reward: 415.08 ± 194.90 (95% CI)


### python test.py -t DQN -a models/agent_DQN_best.pth -s 1 -n 100

Ran 100 episodes.
Average reward: 254.49 ± 159.47 (95% CI)


### python test.py -t ruleBase -s 1 -n 100

Ran 100 episodes.
Average reward: 2564.43 ± 17.60 (95% CI)


# STEP 2 :

### python test.py -t PPO -a models/ppo_sheep1_obst0.zip -s 1 -n 100

Ran 100 episodes.
Average reward: 1641.19 ± 236.84 (95% CI)

### python test.py -t DQN -a models/agent_DQN_best.pth -s 1 -n 100

Ran 100 episodes.
Average reward: 365.53 ± 175.39 (95% CI)

# STEP 3 : 

### python test.py -t PPO -a models/ppo_sheep1_obst2.zip -s 1 -r 0.2 

### python test.py -t PPO -a models/ppo_sheep1_obst2.zip -s 1 -r 0.2 -n 100