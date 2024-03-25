# Pacman Movement Decision System

## Overview
This project implements a decision-making system for Pacman, utilising a combination of Decision Tree and Random Forest classifiers to predict movements within the classic Pacman game environment. The goal is to demonstrate how machine learning models can influence game strategies, albeit with no guarantee of consistently winning games. The unique aspect of this implementation lies in its approach to understanding and acting upon the game's dynamics through learned behaviors from game data.

## Project Structure
The repository contains the `classifier.py` file, which encapsulates the logic for the Decision Tree and Random Forest classifiers. To maintain focus on the AI component and avoid duplicating the entire Berkeley Pacman framework, this repository does not include the Pacman game code. Users interested in testing or further developing the AI must download the Berkeley AI Pacman projects separately and integrate the classifier file into that environment.

### Getting the Pacman Code
1. Visit the [Berkeley AI Pacman Project page](http://ai.berkeley.edu/project_overview.html) to download the required Pacman game files.
2. Extract the files into a suitable directory on your system.

### Integrating the Classifier
After obtaining the Pacman codebase:
1. Place the `classifier.py` file from this repository into the root directory of the extracted Pacman code.
2. Ensure Python 3 is set up on your machine to run the game.

### Running the AI-Controlled Pacman
To launch the game with the AI-controlled Pacman using the implemented classifiers, execute the following command in the terminal within the Pacman directory:
```bash
python3 pacman.py -p ClassifierAgent
