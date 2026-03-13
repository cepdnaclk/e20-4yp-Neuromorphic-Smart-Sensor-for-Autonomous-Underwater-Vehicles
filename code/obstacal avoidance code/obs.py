import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LIF_Neuron:
    """Leaky Integrate-and-Fire Neuron"""
    def __init__(self, threshold=1.0, decay=0.9, refractory_period=5):
        self.threshold = threshold
        self.decay = decay
        self.potential = 0.0
        self.refractory_period = refractory_period
        self.refractory_counter = 0
        self.spike_times = []

    def step(self, input_current, t):
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.potential = 0.0
            return False

        self.potential = self.potential * self.decay + input_current

        if self.potential >= self.threshold:
            self.potential = 0.0
            self.refractory_counter = self.refractory_period
            self.spike_times.append(t)
            return True
        return False

    def reset(self):
        self.potential = 0.0
        self.refractory_counter = 0
        self.spike_times = []

class FourSensorSNN:
    """SNN with 4 Ultrasonic Sensors for Obstacle Avoidance"""
    def __init__(self):
        # Input layer: 4 ultrasonic sensors (front, left, right, back)
        self.sensor_neurons = [LIF_Neuron(threshold=0.7, decay=0.85) for _ in range(4)]

        # Hidden layer: processing neurons for each sensor
        self.processing_neurons = [LIF_Neuron(threshold=0.6, decay=0.9) for _ in range(8)]

        # Smoothing layer (noise filtering)
        self.filter_neurons = [LIF_Neuron(threshold=0.8, decay=0.95) for _ in range(4)]

        # Output neurons (actions)
        self.output_neurons = {
            'move_forward': LIF_Neuron(threshold=0.5, decay=0.9),
            'turn_left': LIF_Neuron(threshold=0.6, decay=0.9),
            'turn_right': LIF_Neuron(threshold=0.6, decay=0.9),
            'slow_down': LIF_Neuron(threshold=0.7, decay=0.9),
            'stop': LIF_Neuron(threshold=0.8, decay=0.9)
        }

        # Weight matrices (to be trained)
        self.w_sensor_to_processing = np.random.randn(4, 8) * 0.3 + 0.5
        self.w_processing_to_filter = np.random.randn(8, 4) * 0.3 + 0.5
        self.w_filter_to_output = np.random.randn(4, 5) * 0.3 + 0.5

        # Learning rate
        self.learning_rate = 0.01

    def distance_to_spike_rate(self, distance, max_distance=200):
        """Convert distance to spike rate (frequency ∝ 1/distance)"""
        if distance < 5:
            distance = 5
        spike_rate = max_distance / distance
        return np.clip(spike_rate / 10, 0, 1.5)

    def forward(self, sensor_data, t):
        """Forward pass through the network"""
        # sensor_data: [front, left, right, back] distances

        # Input layer: convert distances to spike rates
        spike_rates = [self.distance_to_spike_rate(d) for d in sensor_data]
        sensor_spikes = [neuron.step(rate, t) for neuron, rate in zip(self.sensor_neurons, spike_rates)]

        # Hidden layer: processing
        processing_currents = np.zeros(8)
        for i, spike in enumerate(sensor_spikes):
            if spike:
                processing_currents += self.w_sensor_to_processing[i]

        processing_spikes = [neuron.step(current, t) for neuron, current in zip(self.processing_neurons, processing_currents)]

        # Filter layer: smoothing
        filter_currents = np.zeros(4)
        for i, spike in enumerate(processing_spikes):
            if spike:
                filter_currents += self.w_processing_to_filter[i]

        filter_spikes = [neuron.step(current, t) for neuron, current in zip(self.filter_neurons, filter_currents)]

        # Output layer
        output_currents = np.zeros(5)
        for i, spike in enumerate(filter_spikes):
            if spike:
                output_currents += self.w_filter_to_output[i]

        outputs = {}
        output_keys = list(self.output_neurons.keys())
        for i, (key, neuron) in enumerate(self.output_neurons.items()):
            outputs[key] = neuron.step(output_currents[i], t)

        return outputs, spike_rates

    def train_stdp(self, sensor_data, target_action, time_window=50):
        """Train using Spike-Timing-Dependent Plasticity (STDP)"""
        # Reset neurons
        for neuron in self.sensor_neurons + self.processing_neurons + self.filter_neurons:
            neuron.reset()
        for neuron in self.output_neurons.values():
            neuron.reset()

        # Run through time window
        spike_history = []
        for t in range(time_window):
            outputs, _ = self.forward(sensor_data, t)
            spike_history.append(outputs)

        # Calculate reward based on correct action
        reward = 0
        target_spikes = sum([spike_history[t][target_action] for t in range(time_window)])

        if target_spikes > 0:
            reward = 1.0

        # Weight update using reward modulation
        # This is a simplified STDP-like rule
        if reward > 0:
            # Strengthen weights that led to correct action
            self.w_filter_to_output[:, list(self.output_neurons.keys()).index(target_action)] += self.learning_rate * reward
        else:
            # Weaken weights
            self.w_filter_to_output[:, list(self.output_neurons.keys()).index(target_action)] -= self.learning_rate * 0.5

        # Clip weights
        self.w_sensor_to_processing = np.clip(self.w_sensor_to_processing, 0.1, 2.0)
        self.w_processing_to_filter = np.clip(self.w_processing_to_filter, 0.1, 2.0)
        self.w_filter_to_output = np.clip(self.w_filter_to_output, 0.1, 2.0)

        return reward

def generate_training_data(n_samples=1000):
    """Generate synthetic training dataset (replace with real data)"""
    # Format: [front, left, right, back, action]
    # Actions: 0=forward, 1=turn_left, 2=turn_right, 3=slow_down, 4=stop

    data = []
    actions = ['move_forward', 'turn_left', 'turn_right', 'slow_down', 'stop']

    for _ in range(n_samples):
        front = np.random.uniform(5, 200)
        left = np.random.uniform(5, 200)
        right = np.random.uniform(5, 200)
        back = np.random.uniform(5, 200)

        # Decision logic based on sensor readings
        if front < 15 or left < 15 or right < 15:
            action = 'stop'
        elif front < 30:
            if left > right:
                action = 'turn_left'
            else:
                action = 'turn_right'
        elif front < 50 or left < 30 or right < 30:
            action = 'slow_down'
        elif left < 40:
            action = 'turn_right'
        elif right < 40:
            action = 'turn_left'
        else:
            action = 'move_forward'

        data.append([front, left, right, back, action])

    return data

def load_csv_data(filename):
    """Load real-world data from CSV file
    Expected format: front,left,right,back,action
    """
    try:
        data = []
        with open(filename, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split(',')
                front, left, right, back = map(float, parts[:4])
                action = parts[4].strip()
                data.append([front, left, right, back, action])
        print(f"Loaded {len(data)} samples from {filename}")
        return data
    except FileNotFoundError:
        print(f"File {filename} not found. Using synthetic data instead.")
        return None

def train_snn(data, epochs=50):
    """Train the SNN on dataset"""
    snn = FourSensorSNN()

    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    train_rewards = []
    test_accuracies = []

    for epoch in range(epochs):
        epoch_reward = 0

        # Training
        np.random.shuffle(train_data)
        for sample in train_data:
            sensor_readings = sample[:4]
            target_action = sample[4]
            reward = snn.train_stdp(sensor_readings, target_action)
            epoch_reward += reward

        avg_reward = epoch_reward / len(train_data)
        train_rewards.append(avg_reward)

        # Testing
        correct = 0
        for sample in test_data:
            sensor_readings = sample[:4]
            target_action = sample[4]

            # Reset and test
            for neuron in snn.sensor_neurons + snn.processing_neurons + snn.filter_neurons:
                neuron.reset()
            for neuron in snn.output_neurons.values():
                neuron.reset()

            output_counts = {key: 0 for key in snn.output_neurons.keys()}
            for t in range(50):
                outputs, _ = snn.forward(sensor_readings, t)
                for key, spike in outputs.items():
                    if spike:
                        output_counts[key] += 1

            predicted_action = max(output_counts, key=output_counts.get)
            if predicted_action == target_action:
                correct += 1

        accuracy = correct / len(test_data)
        test_accuracies.append(accuracy)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Reward = {avg_reward:.3f}, Test Accuracy = {accuracy:.3f}")

    return snn, train_rewards, test_accuracies

def visualize_training(train_rewards, test_accuracies):
    """Visualize training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_rewards, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.grid(True)

    ax2.plot(test_accuracies, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy Over Time')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def test_real_time(snn, sensor_readings):
    """Test SNN with real-time sensor data"""
    print(f"\nTesting with sensors: Front={sensor_readings[0]:.1f}, Left={sensor_readings[1]:.1f}, "
          f"Right={sensor_readings[2]:.1f}, Back={sensor_readings[3]:.1f}")

    # Reset neurons
    for neuron in snn.sensor_neurons + snn.processing_neurons + snn.filter_neurons:
        neuron.reset()
    for neuron in snn.output_neurons.values():
        neuron.reset()

    output_counts = {key: 0 for key in snn.output_neurons.keys()}

    for t in range(50):
        outputs, _ = snn.forward(sensor_readings, t)
        for key, spike in outputs.items():
            if spike:
                output_counts[key] += 1

    print("Output spike counts:", output_counts)
    action = max(output_counts, key=output_counts.get)
    print(f"Decision: {action.upper()}")
    return action

# Main execution
print("=== SNN Obstacle Avoidance with 4 Ultrasonic Sensors ===\n")

# Try to load real data, otherwise use synthetic
real_data = load_csv_data('sensor_data.csv')
if real_data is None:
    print("Generating synthetic training data...")
    training_data = generate_training_data(1000)
else:
    training_data = real_data

print(f"\nStarting training with {len(training_data)} samples...\n")

# Train the network
trained_snn, rewards, accuracies = train_snn(training_data, epochs=50)

# Visualize results
visualize_training(rewards, accuracies)

# Test with example scenarios
print("\n=== Testing Trained SNN ===")
test_scenarios = [
    [100, 80, 90, 120],   # Clear path
    [20, 60, 70, 100],    # Obstacle in front
    [80, 25, 90, 100],    # Obstacle on left
    [80, 90, 25, 100],    # Obstacle on right
    [10, 15, 12, 80],     # Critical - obstacles everywhere
]

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\nScenario {i}:")
    test_real_time(trained_snn, scenario)

print("\n=== Training Complete ===")
print(f"Final Test Accuracy: {accuracies[-1]:.3f}")
print("\nTo use your own data, create 'sensor_data.csv' with format:")
print("front,left,right,back,action")
print("100,80,90,120,move_forward")
print("20,60,70,100,stop")
print("...")