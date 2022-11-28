# del-image-captioning

Repository für die Minichallenge 2 des Moduls "Deep Learning" im Bachelorstudiengang "Data Science" an der Fachhochschule Nordwestschweiz (FHNW).

Es wurde versucht die Resultate des Papers von [1] zu reproduzieren. 

# Modellarchitektur
![Modellarchitektur](Netzwerkarchitektur-mc2-del.jpg)

# Beschreibung Repository

Um die Notebooks und Dateien ausführen zu können, muss ein Virtualenvironment erstellt und die Abhängigkeiten installiert werden. Diese sind in der Datei *requirements.txt* aufgelistet.

Im Notebook *Modellentwicklung.ipynb* ist die explorative Datenanalyse und die Modellentwicklung beschrieben. In einem weiteren Schritt wurde die Modellstruktur ausgelagert in die Datei *img_cap_lib.py*.

Die beiden Notebooks *training_ConvNeXt_tiny.ipynb* und *training_ResNEXt50.ipynb* beeinhalten die Trainingsprozesse für beide Modelle. 

Die beiden Notebooks *Evaluation_ConvNeXt_tiny.ipynb* und *Evaluation_ResNEXt50.ipynb* beinhalten die Evaluierung der Modelle.

# Quellen

[1] https://arxiv.org/abs/1411.4555