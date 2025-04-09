import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

class DDMSimulator:
    def __init__(self):
        self.setup_params()
        
    def setup_params(self):
        st.title("Drift Diffusion Model Simulator")
        
        # Create two columns for parameters and descriptions
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Model Parameters")
            self.params = {
                'drift_rate': st.slider('Drift Rate', -3.0, 3.0, 1.5, 
                                      help="Average rate of evidence accumulation (higher = faster decisions)"),
                'threshold': st.slider('Threshold', 0.5, 5.0, 2.0,
                                     help="Decision boundary (higher = more accurate but slower)"),
                'bias': st.slider('Starting Bias', -2.0, 2.0, 0.0,
                                help="Starting point of evidence (positive = bias toward upper boundary)"),
                'noise_sd': st.slider('Noise', 0.1, 3.0, 1.0,
                                    help="Standard deviation of noise in evidence accumulation"),
                'max_time': st.slider('Max Time (seconds)', 1.0, 10.0, 3.0,
                                    help="Maximum simulation time"),
                'dt': 0.01
            }
            
        with col2:
            st.subheader("Model Description")
            st.markdown("""
            This simulator shows how decisions might be made in the brain using a 
            Drift Diffusion Model (DDM).
            
            **How it works:**
            1. Evidence accumulates over time (drift rate)
            2. Random noise creates fluctuations
            3. Decision is made when evidence reaches a threshold
            
            **The Plot:**
            - Black line: Evidence accumulation
            - Red line: Upper threshold
            - Blue line: Lower threshold
            - Green dot: Starting point
            """)

    def run_single_trial(self):
        # Initialize arrays for storing the simulation data
        t = np.arange(0, self.params['max_time'], self.params['dt'])
        evidence = np.zeros_like(t)
        evidence[0] = self.params['bias']
        
        # Run the simulation
        decision_made = False
        decision_time = None
        decision_boundary = None
        
        for i in range(1, len(t)):
            # Update evidence
            noise = self.params['noise_sd'] * np.sqrt(self.params['dt']) * np.random.normal()
            evidence[i] = evidence[i-1] + (self.params['drift_rate'] * self.params['dt'] + noise)
            
            # Check for decision
            if abs(evidence[i]) >= self.params['threshold']:
                decision_made = True
                decision_time = t[i]
                decision_boundary = "Upper" if evidence[i] >= self.params['threshold'] else "Lower"
                break
        
        return t, evidence, decision_made, decision_time, decision_boundary

    def plot_trial(self):
        t, evidence, decision_made, decision_time, decision_boundary = self.run_single_trial()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, evidence, 'k-', label='Evidence', linewidth=2)
        ax.axhline(y=self.params['threshold'], color='r', linestyle='-', label='Upper Threshold')
        ax.axhline(y=-self.params['threshold'], color='b', linestyle='-', label='Lower Threshold')
        ax.plot(0, self.params['bias'], 'go', markersize=10, label='Starting Point')
        
        ax.set_xlim(0, self.params['max_time'])
        ax.set_ylim(-self.params['threshold']*1.5, self.params['threshold']*1.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Evidence')
        ax.set_title('Drift Diffusion Model Simulation')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        if decision_made:
            st.success(f"Decision made: {decision_boundary} boundary crossed at {decision_time:.2f} seconds")
        else:
            st.warning("No decision reached within the time limit")

def main():
    st.set_page_config(page_title="DDM Simulator", layout="wide")
    
    simulator = DDMSimulator()
    
    if st.button('Run New Simulation'):
        simulator.plot_trial()

if __name__ == "__main__":
    main() 