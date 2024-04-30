import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Anzahl der Dimensionen
n = 4

# Beispiel-Scores
scores = [5, 7, 3, 9]

# Labels erstellen
labels = ['Dimension {}'.format(i+1) for i in range(4)]
print(labels)
# Horizontaler Balkendiagramm erstellen
plt.barh(np.arange(len(scores)), scores, tick_label=labels)

# Optional: Achsenbeschriftungen hinzufügen
plt.xlabel('Score')
plt.ylabel('Dimension')
print(matplotlib.__version__)
# Plot anzeigen
plt.show()