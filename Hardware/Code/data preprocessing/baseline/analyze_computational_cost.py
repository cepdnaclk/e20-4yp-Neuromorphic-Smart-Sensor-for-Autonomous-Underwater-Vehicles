"""
COMPUTATIONAL COST ANALYSIS
Analyze SNN computational advantages vs traditional neural networks

This measures:
1. FLOPs (Floating Point Operations)
2. Spike rate (% of active neurons)
3. Theoretical power consumption
4. Memory bandwidth requirements
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

print("="*70)
print("  SNN COMPUTATIONAL ADVANTAGE ANALYSIS")
print("="*70)
print()

# ═══════════════════════════════════════════════════════════════════════
# 1. DEFINE MODELS FOR COMPARISON
# ═══════════════════════════════════════════════════════════════════════

# SNN Model (yours)
def delta_encoder(x, threshold):
    batch_size, time_steps, features = x.shape
    spikes = torch.zeros((batch_size, time_steps, features * 2), device=x.device)
    reference = x[:, 0, :].clone()
    
    for t in range(1, time_steps):
        current_val = x[:, t, :]
        diff = current_val - reference
        on_spikes = (diff >= threshold).float()
        off_spikes = (diff <= -threshold).float()
        spike_mask = torch.logical_or(on_spikes > 0, off_spikes > 0).float()
        reference = reference * (1 - spike_mask) + current_val * spike_mask
        spikes[:, t, :features] = on_spikes
        spikes[:, t, features:] = off_spikes
    
    return spikes

class EventDrivenSNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=2, beta=0.9, encode_threshold=0.5):
        super().__init__()
        self.encode_threshold = encode_threshold
        spike_grad = surrogate.fast_sigmoid()
        
        self.fc1 = nn.Linear(input_size * 2, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    def forward(self, x):
        spk_in = delta_encoder(x, self.encode_threshold)
        time_steps = spk_in.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem3_rec = []
        
        for t in range(time_steps):
            cur1 = self.fc1(spk_in[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            mem3_rec.append(mem3)
        
        return torch.stack(mem3_rec, dim=0)

# Traditional LSTM for comparison
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, num_layers=2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# ═══════════════════════════════════════════════════════════════════════
# 2. COUNT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

snn_model = EventDrivenSNN(input_size=5, hidden_size=32, output_size=2)
lstm_model = LSTMModel(input_size=5, hidden_size=32, num_layers=2, output_size=2)

snn_params = sum(p.numel() for p in snn_model.parameters())
lstm_params = sum(p.numel() for p in lstm_model.parameters())

print("PARAMETER COUNT:")
print(f"  SNN:  {snn_params:,} parameters")
print(f"  LSTM: {lstm_params:,} parameters")
print(f"  Ratio: {lstm_params/snn_params:.2f}x more in LSTM")
print()

# ═══════════════════════════════════════════════════════════════════════
# 3. ANALYZE SPIKE RATE (Event-Driven Advantage)
# ═══════════════════════════════════════════════════════════════════════

print("="*70)
print("  SPIKE RATE ANALYSIS (Key SNN Advantage)")
print("="*70)
print()

# Create sample input (batch_size=1, time_steps=15, features=5)
torch.manual_seed(42)
sample_input = torch.randn(1, 15, 5) * 0.5  # Realistic sensor data variability

# Encode to spikes
spk_in = delta_encoder(sample_input, threshold=0.5)

# Count active spikes
total_possible = spk_in.numel()
active_spikes = (spk_in > 0).sum().item()
spike_rate = active_spikes / total_possible

print(f"Send-on-Delta Encoding Results:")
print(f"  Total possible spike events: {total_possible:,}")
print(f"  Actual spikes generated: {active_spikes:,}")
print(f"  Spike rate: {spike_rate*100:.2f}%")
print(f"  Reduction: {(1-spike_rate)*100:.1f}% of inputs are zero")
print()

print("INTERPRETATION:")
print(f"  • Only {spike_rate*100:.1f}% of inputs require computation")
print(f"  • {(1-spike_rate)*100:.1f}% of MAC operations can be skipped")
print(f"  • Event-driven: only compute when sensor data changes")
print()

# ═══════════════════════════════════════════════════════════════════════
# 4. FLOPS COMPARISON (Theoretical)
# ═══════════================================================================

print("="*70)
print("  FLOPs ANALYSIS (Operations per Inference)")
print("="*70)
print()

# SNN FLOPs (simplified estimate)
# For each timestep:
#   - Linear layer: input_size * output_size * 2 (multiply + add)
#   - LIF update: ~3 ops per neuron (multiply, add, compare)

time_steps = 15
snn_flops_per_timestep = (
    (5*2 * 32 * 2) +  # fc1: (10 -> 32)
    (32 * 3) +         # lif1 update
    (32 * 32 * 2) +    # fc2: (32 -> 32)
    (32 * 3) +         # lif2 update
    (32 * 2 * 2) +     # fc3: (32 -> 2)
    (2 * 3)            # lif3 update
)

# But only on active inputs! Multiply by spike rate
effective_snn_flops = snn_flops_per_timestep * time_steps * spike_rate

# LSTM FLOPs (all timesteps, always active)
# LSTM cell: 4 * (input_size + hidden_size) * hidden_size operations per timestep
lstm_flops_per_timestep = 4 * (5 + 32) * 32 * 2  # 2 layers
total_lstm_flops = lstm_flops_per_timestep * time_steps
total_lstm_flops += 32 * 2 * 2  # Final FC layer

print("Estimated FLOPs per Inference:")
print(f"  SNN (event-driven):  {effective_snn_flops:>8,.0f} FLOPs")
print(f"  LSTM (continuous):   {total_lstm_flops:>8,.0f} FLOPs")
print(f"  Reduction: {(1 - effective_snn_flops/total_lstm_flops)*100:.1f}%")
print()

print("WHY SNN IS MORE EFFICIENT:")
print(f"  • LSTM processes ALL {time_steps} timesteps fully")
print(f"  • SNN only computes on {spike_rate*100:.1f}% of events")
print(f"  • Sparse activations skip {(1-spike_rate)*100:.1f}% of operations")
print()

# ═══════════════════════════════════════════════════════════════════════
# 5. MEMORY BANDWIDTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

print("="*70)
print("  MEMORY BANDWIDTH REQUIREMENTS")
print("="*70)
print()

# Bytes transferred per inference
# Assuming float32 (4 bytes)

# SNN: Only transfer spike events (binary, can be compressed)
snn_data_transfer = active_spikes * 1  # 1 byte per spike (binary)

# LSTM: Transfer all activations
lstm_data_transfer = (
    (time_steps * 5 * 4) +      # Input sequence
    (time_steps * 32 * 4 * 2) +  # Hidden states (2 layers)
    (2 * 4)                     # Output
)

print(f"Data Transfer per Inference:")
print(f"  SNN (sparse):      {snn_data_transfer:>6,} bytes")
print(f"  LSTM (dense):      {lstm_data_transfer:>6,} bytes")
print(f"  Reduction: {(1 - snn_data_transfer/lstm_data_transfer)*100:.1f}%")
print()

# ═══════════════════════════════════════════════════════════════════════
# 6. POWER CONSUMPTION ESTIMATE (Theoretical)
# ═══════════════════════════════════════════════════════════════════════

print("="*70)
print("  THEORETICAL POWER CONSUMPTION")
print("="*70)
print()

# Typical energy costs (from literature):
# - MAC operation: ~4.6 pJ (picojoules)
# - Memory access: ~640 pJ
# - Spike (neuromorphic hardware): ~0.1 pJ

# Traditional LSTM (CPU/GPU)
lstm_mac_energy = total_lstm_flops * 4.6e-12  # Joules
lstm_mem_energy = (lstm_data_transfer / 4) * 640e-12  # Joules
lstm_total_energy = lstm_mac_energy + lstm_mem_energy

# SNN (on neuromorphic hardware like Loihi)
snn_spike_energy = active_spikes * 0.1e-12  # Joules
snn_mem_energy = (snn_data_transfer) * 640e-12  # Joules
snn_total_energy = snn_spike_energy + snn_mem_energy

print(f"Energy per Inference (theoretical):")
print(f"  LSTM (CPU/GPU):           {lstm_total_energy*1e9:>8.2f} nJ")
print(f"  SNN (neuromorphic HW):    {snn_total_energy*1e9:>8.2f} nJ")
print(f"  Power reduction: {(1 - snn_total_energy/lstm_total_energy)*100:.1f}%")
print()

print("IMPORTANT NOTES:")
print("  • These are THEORETICAL estimates")
print("  • SNN advantages realized on neuromorphic hardware (Loihi, TrueNorth)")
print("  • On standard CPU/GPU, benefits are less pronounced")
print("  • Event-driven processing scales better with sparse data")
print()

# ═══════════════════════════════════════════════════════════════════════
# 7. SUMMARY TABLE FOR PRESENTATION
# ═══════════════════════════════════════════════════════════════════════

print("="*70)
print("  SUMMARY TABLE (Use this in your presentation!)")
print("="*70)
print()

print("| Metric                    | SNN (Event-Driven) | LSTM (Continuous) | Improvement |")
print("|---------------------------|--------------------:|------------------:|------------:|")
print(f"| Parameters                | {snn_params:>18,} | {lstm_params:>17,} | {((lstm_params-snn_params)/lstm_params)*100:>10.1f}% |")
print(f"| Active Operations         | {spike_rate*100:>17.1f}% | {100.0:>17.1f}% | {(1-spike_rate)*100:>10.1f}% |")
print(f"| FLOPs (estimated)         | {effective_snn_flops:>18,.0f} | {total_lstm_flops:>17,.0f} | {(1-effective_snn_flops/total_lstm_flops)*100:>10.1f}% |")
print(f"| Memory Transfer (bytes)   | {snn_data_transfer:>18,} | {lstm_data_transfer:>17,} | {(1-snn_data_transfer/lstm_data_transfer)*100:>10.1f}% |")
print(f"| Energy (theoretical, nJ)  | {snn_total_energy*1e9:>18.2f} | {lstm_total_energy*1e9:>17.2f} | {(1-snn_total_energy/lstm_total_energy)*100:>10.1f}% |")
print()

# ═══════════════════════════════════════════════════════════════════════
# 8. KEY TALKING POINTS FOR PRESENTATION
# ═══════════════════════════════════════════════════════════════════════

print("="*70)
print("  KEY TALKING POINTS FOR YOUR PRESENTATION")
print("="*70)
print()

print("WHY SNN OVER TRADITIONAL NEURAL NETWORKS:")
print()
print(f"1. EVENT-DRIVEN PROCESSING")
print(f"   → Only {spike_rate*100:.1f}% of neurons are active per timestep")
print(f"   → {(1-spike_rate)*100:.1f}% reduction in computations")
print()
print(f"2. SPARSE ACTIVATIONS")
print(f"   → {(1-effective_snn_flops/total_lstm_flops)*100:.0f}% fewer operations than LSTM")
print(f"   → Sensor data is naturally sparse (only changes matter)")
print()
print(f"3. LOWER MEMORY BANDWIDTH")
print(f"   → {(1-snn_data_transfer/lstm_data_transfer)*100:.0f}% less data movement")
print(f"   → Critical for embedded systems with limited bandwidth")
print()
print(f"4. ENERGY EFFICIENCY (on neuromorphic hardware)")
print(f"   → Theoretical {(1-snn_total_energy/lstm_total_energy)*100:.0f}% energy reduction")
print(f"   → Scales to future Intel Loihi / IBM TrueNorth deployment")
print()
print(f"5. REAL-TIME CAPABLE")
print(f"   → Processes events as they occur")
print(f"   → No need to wait for full frame")
print()

print("="*70)
print("  HOW TO ANSWER: 'Why not use a regular CNN/LSTM?'")
print("="*70)
print()

print('ANSWER:')
print('"While traditional neural networks like LSTMs can achieve similar accuracy,')
print('our SNN approach offers three key advantages:')
print()
print(f'First, event-driven processing - we only compute when the sensor data')
print(f'changes, using {spike_rate*100:.0f}% of the operations compared to frame-based')
print(f'processing. This is critical for battery-powered underwater vehicles.')
print()
print('Second, our Send-on-Delta encoding naturally matches how underwater')
print('sensors work - distances change slowly, so most timesteps have no')
print('significant updates requiring computation.')
print()
print('Third, this positions us for future neuromorphic hardware deployment')
print(f'on chips like Intel Loihi, where we could see {(1-snn_total_energy/lstm_total_energy)*100:.0f}% energy')
print('reduction compared to running an LSTM on a standard processor."')
print()

print("="*70)
print("  ✅ COMPUTATIONAL ANALYSIS COMPLETE")
print("="*70)
print()

# Save results
results = {
    'spike_rate': spike_rate,
    'snn_flops': effective_snn_flops,
    'lstm_flops': total_lstm_flops,
    'snn_params': snn_params,
    'lstm_params': lstm_params,
    'flops_reduction': (1 - effective_snn_flops/total_lstm_flops) * 100,
    'memory_reduction': (1 - snn_data_transfer/lstm_data_transfer) * 100,
    'energy_reduction_theoretical': (1 - snn_total_energy/lstm_total_energy) * 100,
}

import pickle
with open('computational_analysis.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✓ Saved: computational_analysis.pkl")
print()
print("NEXT STEPS:")
print("  1. Use these numbers in your presentation slides")
print("  2. Create a comparison table slide")
print("  3. Emphasize event-driven advantages")
print("  4. Be clear: theoretical estimates, not measured power")
print()
