import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0        # Total time
N = 1000       # Number of steps
dt = T / N     # Time step
t = np.linspace(0, T, N)  # Time grid

# Wiener process (Brownian motion)
W = np.zeros(N)
W[1:] = np.cumsum(np.sqrt(dt) * np.random.randn(N-1))

# Plotting the Wiener process
plt.figure(figsize=(10, 6))
plt.plot(t, W, label='Wiener Process $W(t)$')
plt.title('Wiener Process (Brownian Motion)')
plt.xlabel('Time $t$')
plt.ylabel('W(t)')
plt.grid(True)
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0        # Total time
N = 1000       # Number of steps
dt = T / N     # Time step
t = np.linspace(0, T, N)  # Time grid

# Standard Wiener process (Brownian motion)
W = np.zeros(N)
W[1:] = np.cumsum(np.sqrt(dt) * np.random.randn(N-1))

# Tied-down Wiener process (Brownian bridge)
W_tied_down = W - t * W[-1]

# Plotting the tied-down Wiener process
plt.figure(figsize=(10, 6))
plt.plot(t, W_tied_down, label='Tied-down Wiener Process $W(t)$')
plt.title('Tied-down Wiener Process (Brownian Bridge)')
plt.xlabel('Time $t$')
plt.ylabel('W(t)')
plt.grid(True)
plt.legend()
plt.show()
