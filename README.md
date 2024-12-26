# ASD Diagnosis Support System using Deep Reinforcement Learning

This is a project I built explores the application of DRL to support the diagnosis of Autism Spectrum Disorder (ASD), drawing inspiration from my research on gut dysbiosis and its potential link to ASD.  

Learn more about my research project:
[https://docs.google.com/document/d/e/2PACX-1vT6ieho77aERCnjv_OzUSp_gJtujS5vhUV2_jq2o87hPLqsQZgxed5J90rviF4di2vg82DSf4GLNS3y/pub](https://docs.google.com/document/d/e/2PACX-1vT6ieho77aERCnjv_OzUSp_gJtujS5vhUV2_jq2o87hPLqsQZgxed5J90rviF4di2vg82DSf4GLNS3y/pub)

## Project Goal:

My aim is to develop a DRL agent that learns to predict the probability of ASD based on a set of biomarkers, including gut microbiome composition and other relevant metabolic indicators.

## Data and Biomarkers:

The model is trained on a dataset of patient information (hypothetical in this example, but should be replaced with real data). This data includes:

* **Gut Microbiome Composition:** Relative abundances of specific bacterial species (e.g., Bacteroides, Prevotella, Bifidobacterium). These are represented as numerical features.
* **Metabolic Biomarkers:** Levels of serotonin, propionic acid, neurotoxins, and other relevant metabolites. These are also represented as numerical features.
* **Diagnosis:** A binary variable (0 or 1) indicating the presence or absence of ASD.

## Methodology:

I utilized a DDPG algorithm, a model-free off-policy reinforcement learning approach. The agent learns to map biomarker profiles to a probability of ASD through interaction with an environment.

## Code Structure:

Here are the components of my project

* **`config.py`:** Contains hyperparameters for the DRL agent, such as learning rates, discount factor, replay buffer size, and paths for saving/loading models.
* **`networks.py`:** Defines the neural network architectures for the actor (policy) and critic (Q-function) networks. The actor network takes biomarker profiles as input and outputs a probability of ASD. The critic network evaluates the quality of the actor's actions (predictions). The actor network includes a noisy version for exploration during training.
* **`replay_buffer.py`:** Implements a replay buffer to store experiences (state, action, reward, next state, done) and sample mini-batches for training. This improves learning efficiency and stability.
* **`ddpg_agent.py`:** Implements the DDPG agent. It includes methods for:
    * Initializing the actor and critic networks, their optimizers, and target networks.
    * Updating target networks using soft updates.
    * Generating actions (probabilities of ASD) based on the current state (biomarker profile).
    * Learning from experiences stored in the replay buffer using the DDPG algorithm.
    * Saving and loading trained models.
* **`train.py`:** The main training script. This script:
    * Loads the dataset (placeholder in this example – needs to be replaced with your actual data).
    * Creates an environment that interacts with the agent.
    * Runs the training loop, where the agent interacts with the environment, stores experiences in the replay buffer, and updates its networks.
    * Saves the trained models. This script also includes a placeholder environment (`ASDEnv`) that needs to be replaced with a meaningful environment representing the interaction with the data.


## Mechanisms:

* **Reinforcement Learning:** The agent learns through trial and error. It receives rewards based on how accurately it predicts ASD. The reward function needs careful design to reflect the clinical importance of accurate diagnosis (high rewards for correct predictions, penalties for false positives and negatives).
* **Deep Neural Networks:** Deep neural networks are used to approximate the complex relationship between biomarkers and ASD probability.
* **Replay Buffer:** The replay buffer allows the agent to learn from past experiences, improving sample efficiency and stability.
* **Target Networks:** Target networks are used to stabilize training by providing a slowly updated target for the critic network.

## My Future Development:

- I am currently obtaining and preprocessing a real-world dataset of patient data, including relevant biomarkers and ASD diagnoses.
- After data collection, I will develop a more sophisticated reward function that accurately reflects the clinical importance of accurate diagnosis, based on the results I receive. I will post the newest updates here.
- I will also evaluate the model's performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score, AUC).

You can learn more about my coding projects here:[
https://sites.google.com/view/wong-kin-on-christopher/computer-science](https://sites.google.com/view/wong-kin-on-christopher/computer-science)
