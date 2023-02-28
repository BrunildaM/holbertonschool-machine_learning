#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.title('Number of Fruit per Person')
x = ['Farrah', 'Fred', 'Felicia ']

columns = ('Farrah', 'Fred', 'Felicia ')
rows = ['apples', 'bananas', 'oranges', 'peaches']
colors = ('red', 'yellow', '#ff8000', '#ffe5b4')
n_rows = len(fruit)

index = np.arange(len(columns)) + 0.3
bar_width = 0.5

y_offset = np.zeros(len(columns))

for row in range(n_rows):
        plt.bar(x, fruit[row], bar_width, bottom = y_offset , color = colors[row])
        y_offset = y_offset + fruit[row]

plt.legend(['apples', 'bananas', 'oranges', 'peaches'])
plt.show()
