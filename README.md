# Complement data for: Assessing learning in mosquito larvae using video-tracking

[Martin Dessart](https://martindessart.github.io/), Miguel Piñeirúa, Claudio R. Lazzari, Fernando J. Guerrieri

[Institut de Recherche sur la Biologie de l’Insecte](https://irbi.univ-tours.fr/), UMR7261 CNRS - University de Tours, Tours, France.

# Abstract
Mosquito larvae display a stereotyped escape response when they rest attached to the water surface. It consists in detaching from the surface and diving, to return to the surface after a brief time. It has been shown that this response can be evoked several times, by repeatedly presenting a moving shadow. Diving triggered by a potential danger revealed as a simple bioassay for investigating behavioural responses in mosquito larvae, in particular their ability to learn. In the present work, we describe an automated system, based on video-tracking individuals, and extracting quantitative data of their movements. We validated our system, by reinvestigating the habituation response of larvae of *Aedes aegypti* reared in the laboratory, and providing original data on field-collected larvae of genera *Culex* and *Anopheles*. Habituation could be demonstrated to occur in all the species, even though it was not possible to induce dishabituation in Culex and Anopheles mosquitoes. In addition to non-associative learning, we characterised motor activity in the studied species, thanks to the possibility offered by the tracking system to extract multiple variables. The here-described system and algorithms can be easily adapted to multiple experimental situations and variables of interest.

# Folders

## 0)Video examples
Do not hesitate to download and watch the three edited videos. They represent a training series of ten individuals for the three species studied.
The inter-trial intervals are accelerated. The *Aedes* and *Culex* videos correspond to the experimental group, while the *Anopheles* video corresponds to control n°1 (i.e. with vibration after the tenth trial).

## 1)Tracking programme
Two files used on the videos, extracting individual coordinates and surface.

## 2)Data
All the csv files extracted from the videos and used for the analysis.
1.	Raw: raw files extracted directly from the software. 1 file per experiment (= 10 individuals)
2.	Verified: detection threshold, moving average, low-pass filter, data cleaning
3.	Classified: classification in “Stimulation response” corresponding to data during the stimulus and “Locomotor activity”, corresponding to data during the inter-trial intervals
4.	Grouped: one table per individual

## 3)Analysis
R programs used to answer our questions.
1.	Tracking verification: detection threshold, moving average, low-pass filter, data cleaning. From “Raw” to “Verified” files.
2.	Data classification: classification in “Stimulation response” and “Locomotor activity”. From “Verified” to “Classified” files.
3.	Group data: integrate all data from all experiments to build one dataset per species. From “Classified” to “Grouped” files.
4.	Analyse: code used to analyse learning performance & spontaneous locomotor activity


# Dependencies
* R version 4.1.1 (2021-08-10) -- "Kick Things"
* Python 3.7.11
* Spyder 5.1.5

## Licence
This project is licensed under [MIT Licence](https://opensource.org/license/mit/).
